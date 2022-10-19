""" Contains the l2rpn BaseActionConversion """
import copy

import numpy as np
from abc import ABC, abstractmethod
from grid2op.Action import ActionSpace, PlayableAction
from grid2op.Observation import CompleteObservation
from gym import spaces
from typing import Any

from maze.core.env.action_conversion import ActionConversionInterface, ActionType


def add_auto_line_reconnect_action(state: CompleteObservation, action: PlayableAction) -> PlayableAction:
    """Adds line reconnection action to provided action.

    :param state: The current state of the environment.
    :param action: The action to extend with the automatic line reconnect.
    :return: The original action extended with the automatic line reconnect.
    """

    # get list of offline lines which are not in cooldown
    re_connectable_lines = list(np.nonzero((state.line_status == 0) & (state.time_before_cooldown_line == 0))[0])
    if len(re_connectable_lines):

        # Infer so far impacted power lines
        topological_impact = action.get_topological_impact(state.line_status)
        impacted_power_lines = list(np.where(topological_impact[0])[0])

        # iterate disconnected lines
        for line_idx in re_connectable_lines:

            # it is only allowed to change one line status at a time
            if len(set(impacted_power_lines + [line_idx])) > 1:
                continue

            # check if substations are available
            ex_sub_id = state.line_ex_to_subid[line_idx]
            or_sub_id = state.line_or_to_subid[line_idx]
            subs_in_cooldown = list(np.where(state.time_before_cooldown_sub > 0)[0])
            subs_available = (ex_sub_id not in subs_in_cooldown) and (or_sub_id not in subs_in_cooldown)

            # if conditions are met reconnect line
            if subs_available:
                action = copy.deepcopy(action)
                action.update({"set_line_status": [(line_idx, 1)]})
                break

    return action


class BaseActionConversion(ActionConversionInterface, ABC):
    """Interface specifying the conversion of space to actual environment actions.

    :param action_space: The grid2op action space.
    """

    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space

        self.n_gen = self.action_space.n_gen
        self.n_load = self.action_space.n_load
        self.n_sub = self.action_space.n_sub
        self.n_line = self.action_space.n_line
        self.max_buses = 2
        self.max_links = np.amax(self.action_space.sub_info)

        # generator specific values
        self.max_ramp_down = -self.action_space.gen_max_ramp_down
        self.max_ramp_up = self.action_space.gen_max_ramp_up

    def space_to_maze(self, action: Any, state: CompleteObservation) -> PlayableAction:
        """Converts agent action to environment action.

        :param action: gym space object to parse.
        :param state: the environment state.
        :return: action object.
        """
        raise NotImplementedError

    @abstractmethod
    def maze_to_space(self, action: PlayableAction) -> Any:
        """Converts environment to space action.
        """

    @abstractmethod
    def space(self) -> spaces.Space:
        """Returns respective gym action space.

        :return: Gym action space.
        """

    @staticmethod
    @abstractmethod
    def create_action_hash(action: ActionType) -> int:
        """Create a unique, deterministic int hash of the given maze action.

        :param action: The action that should be hashed.
        :return: An integer action hash.
        """

    @staticmethod
    @abstractmethod
    def inv_action_hash(action_hash: int) -> ActionType:
        """Revert a unique, deterministic int hash to the given maze action.

        :param action_hash: The action created with the self.create_action_hash method.
        :return: The action.
        """

