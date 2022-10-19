"""Redispatching controller running gradient-free black box optimization (cross-entropy method)"""
from typing import Union, Tuple, Dict, Any, Optional, List

import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Action import PlayableAction
from grid2op.Environment.Environment import Environment
from grid2op.Observation import CompleteObservation
from maze.core.annotations import override
from maze.core.utils.factory import ConfigType, Factory

from maze_l2rpn.agents.ce_optimizer.ce_optimizer import logger, CEOptimizer
from maze_l2rpn.agents.ce_optimizer.ce_redispatching_problem import CERedispatchingProblem
from maze_l2rpn.agents.redispatching_controller import RedispatchingController
from maze_l2rpn.env.core_env import L2RPNSeedInfo
from maze_l2rpn.grid_state_observer.grid_state_observer_types import GridElementType
from maze_l2rpn.simulate.grid2op_ca_simulation import grid2op_contingency_analysis_simulate
from maze_l2rpn.space_interfaces.action_conversion.dict_unitary import ActionType
from maze_l2rpn.utils import set_logging_level


class CERedispatchingController(RedispatchingController):
    """Redispatching controller running gradient-free black box optimization (cross-entropy method).

    :param optimizer: The CEOptimizer configuration to be instantiated
    :param problem: Configuration of the problem specification instance, must be derived from CEProblem
    :param logging_level: The logging level to be used for the policy. Options: "debug", "info" or "warning".
    """

    def __init__(self,
                 action_space: ActionSpace,
                 env: Environment,
                 optimizer: ConfigType,
                 problem: ConfigType,
                 logging_level: str):
        super().__init__()

        self.action_space = action_space
        self.env = env

        self.optimizer = Factory(base_type=CEOptimizer).instantiate(optimizer)
        self.problem = Factory(base_type=CERedispatchingProblem).instantiate(problem)

        set_logging_level(logger, logging_level)

    @override(RedispatchingController)
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seed the policy."""

        # Convert to int, if passed as L2RPNSeedInfo.
        if isinstance(seed, L2RPNSeedInfo):
            seed = seed.random_seed

        seed = 123456789
        self.optimizer.rng = np.random.RandomState(seed)

    @override(RedispatchingController)
    def compute_action(self,
                       observation: CompleteObservation,
                       line_contingencies: List[int],
                       lines_to_relieve: List[int],
                       joint_action: Optional[PlayableAction] = None) -> Tuple[ActionType, Dict[str, Any]]:
        """Infers an action by running the cross-entropy optimizer."""
        problem_instance = self.problem.create_instance(self.action_space, observation,
                                                        line_contingencies=line_contingencies,
                                                        lines_to_relieve=lines_to_relieve,
                                                        joint_action=joint_action)
        u_init_loc, u_init_scale, u_min, u_max = problem_instance.get_decision_variables()
        best_u = self.optimizer.optimize(u_init_loc, u_init_scale, u_min, u_max, problem_instance.cost_fn)
        info = {"best_u": best_u}
        return problem_instance.to_action(best_u), info

    def perform_contingency_analysis(self, state: CompleteObservation, action: PlayableAction,
                                     ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Simulates the given action and performs analysis of base case and contingency violations.

        :return A tuple (base max rho, line contingencies, lines to relieve)
        """
        line_contingencies = []
        lines_to_relieve = []

        # only simulate base case if needed
        base_case = None

        for contingency in self.problem.contingencies:
            assert len(contingency) == 1, "only n-1 contingencies supported"
            line = contingency[0]
            assert line.element_type == GridElementType.line, "only line contingencies supported"

            if state.line_status[line.idx] == 0:
                continue

            # todo: make configurable
            if line.idx in [93, 180]:
                if base_case is None:
                    base_case, _, _, _ = grid2op_contingency_analysis_simulate(
                        state, action,
                        ignore_hard_overflow=True,
                        use_forecast=True)

                lines_to_relieve.append((line.idx, base_case.rho[line.idx]))
                continue

            action_dict = contingency.to_action_dict(state)

            playable_contingency_action = self.env.action_space(action_dict)
            assert isinstance(playable_contingency_action, PlayableAction)

            obs, _, done, info = grid2op_contingency_analysis_simulate(
                state, playable_contingency_action + action,
                ignore_hard_overflow=True,
                use_forecast=True)

            obs_rho_max = np.inf if done else obs.rho.max()
            line_contingencies.append((contingency, obs_rho_max))

        return line_contingencies, lines_to_relieve
