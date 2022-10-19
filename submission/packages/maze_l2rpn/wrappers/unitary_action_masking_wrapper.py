"""Contains a wrapper for action masking when using the unitary action_conversion"""
from abc import ABC
from typing import Dict, Union, Optional, Any, Tuple

import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces
from maze.core.annotations import override
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.log_stats.event_decorators import define_step_stats, define_epoch_stats, define_episode_stats
from maze.core.wrappers.wrapper import ObservationWrapper

from maze_l2rpn.env.maze_env import Grid2OpEnvironment


class UnitaryActionMaskingEvents(ABC):
    """
    Event topic class with logging statistics for the Unitary action masking wrapper.
    """

    @define_epoch_stats(np.mean)
    @define_episode_stats(np.mean)
    @define_step_stats(np.mean)
    def percent_of_actions_masked_out(self, value: float):
        """The percent of actions being masked out by the wrapper.

        :param value: The percent of actions being masked out at the current step.
        """


class UnitaryActionMaskingWrapper(ObservationWrapper[Grid2OpEnvironment]):
    """Wrapper for masking out actions of the unitary action space. The action masking is performed based on
    simple checks whether the actions will change anything or will result in an illegal action.

    :param env: Environment to wrap.
    :param mask_out_illegal_actions: Specify whether to mask out illegal actions.
    :param check_topo_change: It True check if actions actually change the topology.
    """

    def __init__(self,
                 env: MazeEnv,
                 mask_out_illegal_actions: bool,
                 check_topo_change: bool):
        super().__init__(env)

        # Assert that the unitary action space is used.
        self._action_masking_events = self.core_env.context.event_service.create_event_topic(UnitaryActionMaskingEvents)

        # Set parameters.
        self._mask_out_illegal_actions = mask_out_illegal_actions
        self._check_topo_change = check_topo_change
        self._n_actions = self.action_conversion.num_active_actions

        # The action object can infer the topological impact and will return an array. The first
        #   n_line elements of the array are the status of the power-lines followed by the substation information.
        #   The substation information consists of one element for each bus connection specifying to which bus the
        #   action will try to switch the connection. In order to speed up the processing we infer the positions of each
        #   substation in the array at this point.
        self._sub_to_action_vec_pos = dict()
        count = 0
        for idx, num_connections in enumerate(env.wrapped_env.sub_info):
            num_connections = int(num_connections)
            self._sub_to_action_vec_pos[idx] = (count, count + num_connections)
            count = count + num_connections

        # Update observation space
        assert len(self.observation_spaces_dict.keys()) == 1
        step_key = list(self.observation_spaces_dict.keys())[0]
        self._observation_spaces_dict = self.env.observation_spaces_dict
        self._observation_spaces_dict[step_key] = spaces.Dict({
            **self.env.observation_spaces_dict[step_key].spaces,
            'action_mask': spaces.Box(dtype=np.float32, shape=(self._n_actions,),
                                      low=np.float32(0), high=np.float32(1)),
        })

        self._current_illegal_actions = np.zeros(self._n_actions)

    @override(ObservationWrapper)
    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply substation mask to original maks.

        :param observation: The observation to be updated.
        :return: The updated observation.
        """
        self._current_illegal_actions = np.zeros(self._n_actions)
        action_mask = self._build_action_mask()
        if "action_mask" in observation:
            observation["action_mask"][action_mask == 1.0] = 1.0
        else:
            observation['action_mask'] = action_mask
        return observation

    def clone_from(self, env: 'UnitaryActionMaskingWrapper') -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`."""
        self._n_actions = env._n_actions
        self._sub_to_action_vec_pos = env._sub_to_action_vec_pos
        self._current_illegal_actions = env._current_illegal_actions.copy()
        self.env.clone_from(env)

    def _build_action_mask(self) -> np.ndarray:
        """Prepare mask considering only links related to certain substations.

        :return: The link mask.
        """
        state: CompleteObservation = self.get_maze_state()
        mask = np.ones(self._n_actions, dtype=np.float32)

        # early masking of substations in cooldown
        for sub_id in np.where(state.time_before_cooldown_sub > 0)[0]:
            mask[self.env.action_conversion.substations == sub_id] = 0.0

        # iterate over all actions not yet masked out
        valid_candidates = np.where(mask)[0][1:]
        lines_going_into_maintenance = np.where(state.time_next_maintenance == 1)[0]
        for idx in valid_candidates:
            # Retrieve action
            playable_action = self.action_conversion.space_to_maze({'action': idx}, state=state)

            # Infer the impacted topology components
            topological_impact = playable_action.get_topological_impact(state.line_status)
            impacted_substations = np.where(topological_impact[1])[0]
            impacted_power_lines = np.where(topological_impact[0])[0]
            assert len(impacted_substations) <= 1

            # Infer illegal actions:
            illegal_substation_cooldown = any(map(lambda x: state.time_before_cooldown_sub[x] > 0,
                                                  impacted_substations))
            illegal_powerline_cooldown = any(map(lambda x: state.time_before_cooldown_line[x] > 0,
                                                 impacted_power_lines))
            self._current_illegal_actions[idx] = illegal_powerline_cooldown or illegal_substation_cooldown

            if self._mask_out_illegal_actions and self._current_illegal_actions[idx]:
                mask[idx] = 0.0
                continue

            # Infer actions with line maintenance conflicts
            if len(lines_going_into_maintenance) > 0 and len(impacted_substations) > 0:
                for line_id in lines_going_into_maintenance:
                    sub_id_or = state.line_or_to_subid[line_id]
                    sub_id_ex = state.line_ex_to_subid[line_id]
                    if sub_id_or in impacted_substations or sub_id_ex in impacted_substations:
                        mask[idx] = 0.0
                        break

                # no need to continue
                if mask[idx] == 0:
                    continue

            # Infer noop actions
            if self._check_topo_change:
                predicted_change = False
                action_target_topology = playable_action._get_array_from_attr_name('_set_topo_vect')
                for i in impacted_substations:
                    current_sub_topology = state.sub_topology(i)
                    start_idx, end_idx = self._sub_to_action_vec_pos[i]
                    sub_target_topology = action_target_topology[start_idx: end_idx]
                    predicted_change = predicted_change or np.any(sub_target_topology != current_sub_topology)
                mask[idx] = float(predicted_change)

        percent_of_masked_out_actions = 1 - np.count_nonzero(mask) / float(len(mask))
        self._action_masking_events.percent_of_actions_masked_out(percent_of_masked_out_actions)
        return mask.astype(np.float32)

    @property
    @override(StructuredEnvSpacesMixin)
    def observation_spaces_dict(self) -> Dict[Union[int, str], spaces.Dict]:
        """Implementation of :class:`~maze.core.env.structured_env_spaces_mixin.StructuredEnvSpacesMixin` interface.
        """
        return self._observation_spaces_dict

    @override(ObservationWrapper)
    def get_observation_and_action_dicts(self, maze_state: Optional[MazeStateType],
                                         maze_action: Optional[MazeActionType],
                                         first_step_in_episode: bool) \
            -> Tuple[Optional[Dict[Union[int, str], Any]], Optional[Dict[Union[int, str], Any]]]:
        """Convert the observations, keep actions the same."""
        # requires to have the state set
        self.env.set_maze_state(maze_state)
        return super().get_observation_and_action_dicts(maze_state, maze_action, first_step_in_episode)
