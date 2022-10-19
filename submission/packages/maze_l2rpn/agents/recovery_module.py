"""Contains the recovery module for the maze submission."""
import time
from typing import Tuple, Dict

import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation

from maze_l2rpn.env.maze_env import Grid2OpEnvironment
from maze_l2rpn.simulate.grid2op_simulate import grid2op_simulate

import logging
logger = logging.getLogger('RECOVERY   ')
logger.setLevel(logging.INFO)

class RecoveryModule:
    """The recovery module responsible for recovering to the recovery topology.

    :param submission_params: The submission parameters.
    :param maze_env: The maze env.
    """
    def __init__(self, submission_params: 'SubmissionParams', maze_env: Grid2OpEnvironment):
        self._submission_params = submission_params
        self._recovery_topology = submission_params.RECOVERY_TOPO_VEC
        self._with_recovery = submission_params.WITH_RECOVERY
        self._recovery_max_rho = submission_params.RECOVERY_MAX_RHO
        self._recovery_rho_safe = submission_params.RECOVERY_RHO_SAFE
        self._recovery_check_full_recovery = submission_params.CHECK_FULL_RECOVERY
        self._prepare_recovery_topology(sub_info=maze_env.wrapped_env.sub_info)
        self._recovery_topo_vect = None

    def _prepare_recovery_topology(self, sub_info: Dict[int, int]) -> None:
        """Prepare recovery topology configuration on substation level.

        :param sub_info: Substation info from the grid2op env (number of links per substation)
        """
        if self._recovery_topology is None:
            self._recovery_topology = dict()
        else:
            self._recovery_topology = dict(self._recovery_topology)

        self._recovery_topo_vect = []
        for sub_id in range(len(sub_info)):
            n_links = sub_info[sub_id]

            if sub_id not in self._recovery_topology:
                self._recovery_topology[sub_id] = [1] * n_links
            else:
                assert len(self._recovery_topology[sub_id]) == n_links
                for bus in self._recovery_topology[sub_id]:
                    assert bus in [1, 2]

            # collect recovery topo vector entries
            self._recovery_topo_vect.append(self._recovery_topology[sub_id])

        # convert to array
        self._recovery_topo_vect = np.concatenate(self._recovery_topo_vect)

    @staticmethod
    def sub_topology(maze_state: CompleteObservation, sub_id: int):
        """
        Returns the topology of the given substation
        Returns
        -------

        """
        if hasattr(maze_state, 'sub_topology'):
            sub_topology = maze_state.sub_topology(sub_id)
        else:
            topo_vect_to_sub = np.repeat(np.arange(maze_state.n_sub), repeats=maze_state.sub_info)
            sub_topology = maze_state.topo_vect[topo_vect_to_sub == sub_id]
            sub_topology.flags.writeable = False
        return sub_topology

    def compute_recovery_action(self, env: Grid2OpEnvironment, maze_state: CompleteObservation) -> Tuple[
        PlayableAction, bool]:
        """Compute the recovery action.

        :param env: The maze env.
        :param maze_state: Complete observation
        :return: A playable recovery or noop action.
        """
        start_time = time.time()
        action = env.action_conversion.noop_action()
        action_is_noop = True

        # check if a full recovery would be beneficial
        full_recovery_beneficial = False
        if self._recovery_check_full_recovery:
            full_recovery_beneficial = self.check_full_recovery(maze_env=env)

        # collect actions to take
        actions_to_take = []
        for sub_id in range(maze_state.n_sub):
            if not np.array_equal(self.sub_topology(maze_state, sub_id), self._recovery_topology[sub_id]):
                action_dict = dict()
                action_dict["set_bus"] = {"substations_id": [(sub_id, self._recovery_topology[sub_id])]}
                actions_to_take.append(action_dict)

        # select next
        least_rho_sim = np.inf
        least_rho_state = maze_state.rho.max()
        for action_dict in actions_to_take:
            playable_action = env.wrapped_env.action_space(action_dict)
            sim_state, _, done, _ = grid2op_simulate(maze_state, playable_action,
                                                     use_forcast=True, post_state_preparation=False,
                                                     assert_env_dynamics=False)
            sim_max_rho = sim_state.rho.max()
            if not done and sim_max_rho < self._recovery_max_rho \
                    and (sim_max_rho < (1.025 * least_rho_state) or
                         full_recovery_beneficial or
                         sim_max_rho < self._recovery_rho_safe):
                # take best recovery actions first
                if sim_max_rho < least_rho_sim:
                    least_rho_sim = sim_max_rho
                    action = playable_action
                    action_is_noop = False

        if not action_is_noop:
            substation = int(np.where(playable_action.get_topological_impact(maze_state.line_status)[1])[0][0])
            logger.info(
                f'[{maze_state.current_step:4}] [{"RECOVERY":14}] @sub:{substation:3}, '
                f'n_simulations {len(actions_to_take):4}, time: {(time.time() - start_time):.3f}, '
                f'lookahead {1}, [{least_rho_sim}]')

        return action, action_is_noop

    def has_recovered(self, state: CompleteObservation) -> bool:
        """Returns true if recovery topology hase been reached.

        :return: Recovery status.
        """
        return np.all(self._recovery_topo_vect == state.topo_vect)

    def recovery_possible(self, maze_env: Grid2OpEnvironment) -> bool:
        """Checks if recovery is possible.

        :return: True if recovery is possible.
        """
        state = maze_env.get_maze_state()

        # do not recover if lines are offline
        if np.any(state.line_status == 0):
            return False

        # do not recover if max rho is too high
        if state.rho.max() > self._recovery_max_rho:
            return False

        # do not recover in critical states
        if not maze_env.observation_conversion.is_safe_state(state):
            return False

        # check if maintenance will take place
        n_sub_station_actions_required = \
            np.count_nonzero(
                [np.any(self.sub_topology(state, sub_id) != self._recovery_topology[sub_id])
                 for sub_id in range(state.n_sub)])
        if np.count_nonzero((state.time_next_maintenance <= n_sub_station_actions_required) &
                            (state.time_next_maintenance >= 0)) > 0:
            return False

        # check if substations are not in cooldown
        for sub_id in range(state.n_sub):
            if not np.all(self.sub_topology(state, sub_id) == self._recovery_topology[sub_id]) \
                    and state.time_before_cooldown_sub[sub_id] > 0:
                return False

        return True

    def check_full_recovery(self, maze_env: Grid2OpEnvironment) -> bool:
        """Check if full recovery would be beneficial.

        :return: True if beneficial, else False.
        """
        # get current maze state
        state: CompleteObservation = maze_env.get_maze_state()

        # prepare complete recovery action
        action = maze_env.wrapped_env.action_space({})
        set_bus = np.full((state.dim_topo,), fill_value=0, dtype=state.topo_vect.dtype)
        set_bus[state.topo_vect != 1] = 1
        action.set_bus = set_bus

        # fix simulation env parameters two allow to modify more than one substation
        org_MAX_LINE_STATUS_CHANGED = state._obs_env._parameters.MAX_SUB_CHANGED
        state._obs_env._parameters.MAX_SUB_CHANGED = state.n_sub
        sim_state, sim_rew, sim_done, sim_dict = grid2op_simulate(state, action,
                                                                  use_forcast=True, post_state_preparation=False,
                                                                  assert_env_dynamics=False)

        # reset simulation env parameters
        state._obs_env._parameters.MAX_SUB_CHANGED = org_MAX_LINE_STATUS_CHANGED

        # check if recovery makes sense
        cur_max_rho = state.rho.max()
        exp_max_rho = sim_state.rho.max()
        beneficial = exp_max_rho < (cur_max_rho * 1.025)

        return beneficial
