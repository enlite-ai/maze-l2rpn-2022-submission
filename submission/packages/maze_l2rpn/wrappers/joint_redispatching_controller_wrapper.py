"""Wrapper that integrates a redispatching controller for control of redispatching,
curtailment and storages."""
import logging
import time
from typing import Any, Union

import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.utils.factory import ConfigType, Factory
from maze.core.wrappers.wrapper import ObservationWrapper, ActionWrapper

from maze_l2rpn.agents.redispatching_controller import RedispatchingController
from maze_l2rpn.agents.ce_redispatching_controller import CERedispatchingController
from maze_l2rpn.env.core_env import L2RPNSeedInfo
from maze_l2rpn.env.maze_env import Grid2OpEnvironment
from maze_l2rpn.simulate.grid2op_simulate import grid2op_simulate

logger = logging.getLogger('REDISP_CONT')
logger.setLevel(logging.INFO)


class JointRedispatchingControllerWrapper(ActionWrapper[Grid2OpEnvironment]):
    """Wrapper that integrates a redispatching controller for control of redispatching,
    curtailment and storages into the env.

    :param redispatching_controller: The redispatching controller to use.
    :param joint_action_type: Specifies how the joint topology redispatching action is combined ["concat", "joint"].
    """

    def __init__(self,
                 env: MazeEnv,
                 redispatching_controller: Union[RedispatchingController, ConfigType],
                 contingency_redispatching_controller: Union[None, RedispatchingController, ConfigType],
                 joint_action_type: str,
                 action_candidates: int):
        super().__init__(env)

        self.redispatching_controller = Factory(RedispatchingController).instantiate(
            redispatching_controller,
            action_space=self.env.action_conversion.action_space,
            env=self.env.wrapped_env
        )

        self.contingency_redispatching_controller = Factory(RedispatchingController).instantiate(
            contingency_redispatching_controller,
            action_space=self.env.action_conversion.action_space,
            env=self.env.wrapped_env
        )

        assert joint_action_type in ["concat", "joint"]
        self.joint_action_type = joint_action_type
        self.action_candidates = action_candidates

        # call line relieve optimization less often by keeping track of the last rho values and only running
        # the optimization if the delta is above a certain threshold
        self.last_contingency_optimized_rho = None

        # stores topo actions to try with redispatching
        self.last_top_rhos = []
        self.last_top_maze_actions = []

    def act_in_safe_state(self, state: CompleteObservation, topo_action: PlayableAction) -> PlayableAction:
        """Perform contingency analysis and optimize for contingency violations."""
        topo_max_rho = -1.0
        lines_to_relieve = []
        contingency_violations = []
        contingency_violations_after = []
        start_time = time.time()
        try:
            # safe state, check if topo_action is no-op
            if topo_action.impact_on_objects()["has_impact"]:
                # recovery action identified, don't perform contingency analysis
                selected_action_type = "topo-recovery"
                return topo_action

            # we are safe and the topology action is no-op -> treat contingencies
            assert isinstance(self.redispatching_controller, CERedispatchingController)
            ca_result_topo = self.redispatching_controller.perform_contingency_analysis(state, topo_action)
            line_contingencies, lines_to_relieve = ca_result_topo
            # consider everything above 190% a contingency violation
            contingency_violations = [c for c in line_contingencies if c[1] >= 1.9]

            if len(lines_to_relieve) == 0 and not contingency_violations:
                # no contingency violations, we can skip the optimizer
                selected_action_type = "noop"
                return topo_action

            if self.last_contingency_optimized_rho is not None and not contingency_violations:
                # no hard contingency violation, see if the rho values of the lines to relieve are
                # good enough and can be skipped
                skip_redispatching = True
                for line, line_rho in lines_to_relieve:
                    # allow an increase of 5 percentage points before triggering a new redispatching optimizer run
                    if line_rho > self.last_contingency_optimized_rho[line] + 0.05:
                        skip_redispatching = False

                    self.last_contingency_optimized_rho[line] = min(self.last_contingency_optimized_rho[line], line_rho)

                if skip_redispatching:
                    selected_action_type = "noop"
                    return topo_action

            # if we have a separate redispatching controller for contingencies, use it,
            # otherwise fall back to the primary redispatching controller
            if self.contingency_redispatching_controller:
                contingency_redisp_controller = self.contingency_redispatching_controller
            else:
                contingency_redisp_controller = self.redispatching_controller

            contingency_redisp_action, redisp_info = contingency_redisp_controller.compute_action(
                observation=state,
                line_contingencies=[c[0] for c in line_contingencies],
                lines_to_relieve=[line[0] for line in lines_to_relieve],
                joint_action=topo_action)
            assert isinstance(contingency_redisp_action, PlayableAction)

            if len(lines_to_relieve):
                # update the `last_contingency_optimized_rho` values
                state_after_redisp, _, done_after_redisp, _ = grid2op_simulate(
                    state=state,
                    playable_action=contingency_redisp_action + topo_action,
                    use_forcast=True,
                    post_state_preparation=False,
                    assert_env_dynamics=False)
                self.last_contingency_optimized_rho = state_after_redisp.rho.copy()

            if len(contingency_violations):
                # test impact of ca redispatch action
                ca_result_after = self.redispatching_controller.perform_contingency_analysis(state,
                                                                                             contingency_redisp_action)
                after_line_contingencies, after_lines_to_relieve = ca_result_after
                contingency_violations_after = [c for c in after_line_contingencies if c[1] >= 1.9]
                cv_dict = dict(contingency_violations)
                cv_dict_after = dict(contingency_violations_after)
                for line, load in cv_dict.items():
                    if line in cv_dict_after and cv_dict_after[line] > load:
                        selected_action_type = "ca_failed"
                        return topo_action

            selected_action_type = "ca_redisp"
            return contingency_redisp_action + topo_action

        finally:
            contingency_violations_str = ",".join([f"({grid_element[0].idx},{max_rho:.3f})"
                                                   for grid_element, max_rho in contingency_violations])
            contingency_violations_after_str = ",".join([f"({grid_element[0].idx},{max_rho:.3f})"
                                                   for grid_element, max_rho in contingency_violations_after])
            lines_to_relieve_str = ",".join([f"({line},{rho:.3f})" for line, rho in lines_to_relieve])

            logger.log(logging.DEBUG if selected_action_type == "noop" else logging.INFO,
                       msg=f'[{state.current_step:4}] [{"ACTION TYPE":14}] [{selected_action_type[:10].upper():10}] '
                           f'max_rho after topo: {topo_max_rho:.3f}, '
                           f'contingency violations: [{contingency_violations_str}], '
                           f'contingency violations after: [{contingency_violations_after_str}], '
                           f'lines_to_relieve: [{lines_to_relieve_str}], '
                           f'elapsed: {(time.time() - start_time):.3f}')

    def action(self, action: ActionType) -> PlayableAction:
        """Apply the redispatching controller policy on top of the provided action and step the env."""
        start_time = time.time()
        state: CompleteObservation = self.env.get_maze_state()
        topo_action: PlayableAction = self.env.action_conversion.space_to_maze(action, state)

        if self.env.observation_conversion.grid_state_observer \
                and self.env.observation_conversion.grid_state_observer.is_state_safe(state):
            start_safe_time = time.time()
            action = self.act_in_safe_state(state, topo_action)
            end_time = time.time()
            self.env.core_env.profiling_events.meta_controller_step_time(total=end_time-start_time,
                                                                         unsafe=0, safe=end_time-start_safe_time)

            return action

        start_unsafe_time = time.time()
        action = self.act_in_unsafe_state(state, topo_action)
        end_time = time.time()
        self.env.core_env.profiling_events.meta_controller_step_time(total=end_time-start_time,
                                                                     unsafe=end_time-start_unsafe_time, safe=0)

        return action

    def act_in_unsafe_state(self, state: CompleteObservation, topo_action: PlayableAction) -> PlayableAction:
        # Get action from the redispatching controller
        start_act_time = time.time()
        redisp_action, redisp_info = self.redispatching_controller.compute_action(state,
                                                                                  line_contingencies=[],
                                                                                  lines_to_relieve=[],
                                                                                  joint_action=None)

        start_sim_time = time.time()
        state_after_topo, _, done_after_topo, _ = grid2op_simulate(
            state=state,
            playable_action=topo_action,
            use_forcast=True,
            post_state_preparation=False,
            assert_env_dynamics=False)
        max_rho_after_topo = state_after_topo.rho.max() if not done_after_topo else np.inf

        state_after_redisp, _, done_after_redisp, _ = grid2op_simulate(
            state=state,
            playable_action=redisp_action,
            use_forcast=True,
            post_state_preparation=False,
            assert_env_dynamics=False)
        max_rho_after_redisp = state_after_redisp.rho.max() if not done_after_redisp else np.inf
        end_sim_time = time.time()

        # prefer redispatching over topology, consider topology only if we see critical line overloads
        if max_rho_after_redisp <= max_rho_after_topo or max_rho_after_redisp <= 0.99:
            selected_action = redisp_action
            selected_action_type = "redispatch"
        else:
            selected_action = topo_action
            selected_action_type = "topology"

        # test joint action
        min_max_rho = min(max_rho_after_topo, max_rho_after_redisp)
        redisp_actions_joint, max_rhos_after_joint, best_us = [], [], []
        if min_max_rho > 0.99:

            # iterate top topology action candidates
            for topo_action_cand in self.last_top_maze_actions:

                # re-use pre-computed redispatch action
                if self.joint_action_type == "concat":
                    redisp_action_cand = redisp_action
                # compute redispatch on top of topology action
                else:
                    redisp_action_cand, redisp_info = \
                        self.redispatching_controller.compute_action(state,
                                                                     line_contingencies=[],
                                                                     lines_to_relieve=[],
                                                                     joint_action=topo_action_cand)

                # compile and try joint action
                joint_action = topo_action_cand + redisp_action_cand
                state_after_joint, _, done_after_joint, _ = grid2op_simulate(
                    state=state,
                    playable_action=joint_action,
                    use_forcast=True,
                    post_state_preparation=False,
                    assert_env_dynamics=False)
                max_rho_after_joint = state_after_joint.rho.max() if not done_after_joint else np.inf

                # bookkeeping
                max_rhos_after_joint.append(max_rho_after_joint)
                redisp_actions_joint.append(redisp_action_cand)

            # select joint action
            if len(max_rhos_after_joint) > 0:
                idx_min = np.argmin(max_rhos_after_joint)
                if max_rhos_after_joint[idx_min] < min_max_rho:
                    selected_action = redisp_actions_joint[idx_min] + self.last_top_maze_actions[idx_min]
                    selected_action_type = self.joint_action_type

        # Log action type and stats
        max_rhos_after_joint = ", ".join(f"{r:.3f}" for r in max_rhos_after_joint)
        logger.info(f'[{state.current_step:4}] [{"ACTION TYPE":14}] [{selected_action_type[:10].upper():10}] '
                    f'max_rho after topo: {max_rho_after_topo:.3f}, '
                    f'max_rho after redisp: {max_rho_after_redisp:.3f}, '
                    f'max_rho after joint: [{max_rhos_after_joint}], '
                    f'redisp_act_time: {(start_sim_time - start_act_time):.3f}, '
                    f'sim_time: {(end_sim_time - start_sim_time):.3f}')

        return selected_action

    @override(ObservationWrapper)
    def reset(self) -> Any:
        """Default reset behavior"""
        self.last_top_rhos = []
        self.last_top_maze_actions = []
        return self.env.reset()

    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seeds the random generator of the controller."""
        self.redispatching_controller.seed(seed)
        self.env.seed(seed)

    @override(ObservationWrapper)
    def clone_from(self, env: 'JointRedispatchingControllerWrapper') -> None:
        """No state required"""
        self.env.clone_from(env)

    def reverse_action(self, action: Any) -> Any:
        """Action reversal not possible"""
        raise NotImplementedError

    def append_top_actions(self, risk: float, maze_action: PlayableAction) -> None:
        """Maintains a list of topology action candidates to try in combination with redispatching.

        :param risk: Risk (e.g., max rho) corresponding to the respective action.
        :param maze_action: The maze action to append.
        """
        top = self.action_candidates
        if len(self.last_top_rhos) < top or risk < np.max(self.last_top_rhos):
            self.last_top_rhos.append(risk)
            self.last_top_maze_actions.append(maze_action)

            if len(self.last_top_rhos) > top:
                idxs = np.argsort(self.last_top_rhos)[:top]
                self.last_top_rhos = [self.last_top_rhos[j] for j in idxs]
                self.last_top_maze_actions = [self.last_top_maze_actions[j] for j in idxs]
