"""Maze Link Prediction ActionConversion interface."""
import logging
import os
from abc import ABC
from typing import Dict, Union, List, Optional

import grid2op
import numpy as np
from grid2op.Action import TopologyAndDispatchAction, PlayableAction
from grid2op.Converter import IdToAct
from grid2op.Observation import CompleteObservation
from gym import spaces
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface

from maze_l2rpn.space_interfaces.action_conversion.base import add_auto_line_reconnect_action

logger = logging.getLogger('ACT CONV   ')
logger.setLevel(logging.INFO)

ActionType = Dict[str, Union[int, List[float]]]


class ActionConversion(ActionConversionInterface, ABC):
    """Interface specifying the conversion of space to actual environment actions.

    :param grid2op_env: The grid2op environment.
    :param action_selection_vector_dump: Path to an .npy file holding the action_selection_vector.
    """

    def __init__(self, grid2op_env: grid2op.Environment, action_selection_vector_dump: Optional[str]):
        super().__init__()
        self.action_space = grid2op_env.action_space

        self.n_gen = self.action_space.n_gen
        self.n_load = self.action_space.n_load
        self.n_sub = self.action_space.n_sub
        self.n_line = self.action_space.n_line

        self.id_to_act = IdToAct(self.action_space)
        self.id_to_act.init_converter(set_line_status=False,
                                      change_line_status=False,
                                      set_topo_vect=True,
                                      change_bus_vect=False,
                                      redispatch=False,
                                      curtail=False,
                                      storage=False)

        self.auto_reconnect_lines = True

        # load masking vector if dump file was provided
        if action_selection_vector_dump:
            assert os.path.exists(action_selection_vector_dump), "Action conversion: Could not find action selection " \
                                                                 "vector dump file."
            action_selection_vector = np.load(action_selection_vector_dump)

        # prepare masking vector
        if action_selection_vector is None:
            action_selection_vector = np.asarray(range(self.id_to_act.n), dtype=np.int)

        # sort, so that noop is still the first action
        action_selection_vector.sort()
        # check that the noop is actually present (in case we have a reduced action space)
        if action_selection_vector[0] != 0:
            print("No-op action (0) was not present in the action selection vector!"
                  " No-op action added at index zero.")
            action_selection_vector = np.concatenate([[0], action_selection_vector])

        # pre-compute required mappings
        self.num_active_actions = len(action_selection_vector)

        self._topo_hash_to_converter_id = dict()
        self._space_id_to_converter_id = dict()
        self._converter_id_to_space_id = dict()
        for i, converter_id in enumerate(action_selection_vector):
            self._space_id_to_converter_id[i] = converter_id
            self._converter_id_to_space_id[converter_id] = i

            # preserve action topo hash
            action_obj = self.id_to_act.convert_act(converter_id)
            self._topo_hash_to_converter_id[tuple(action_obj.set_bus)] = converter_id

        # compute affected substations
        self.substations = []
        for i, converter_id in enumerate(action_selection_vector):
            action_obj = self.id_to_act.convert_act(converter_id)
            sub_id = np.unique(self.action_space._topo_vect_to_sub[np.where(action_obj.set_bus != 0)])
            assert len(sub_id) <= 1
            sub_id = sub_id[0] if len(sub_id) == 1 else -1
            self.substations.append(sub_id)
        self.substations = np.asarray(self.substations, dtype=np.int)

    @override(ActionConversionInterface)
    def space_to_maze(self, action: ActionType, state: Optional[CompleteObservation], is_simulation: bool = False) \
            -> TopologyAndDispatchAction:
        """Converts space to environment action.
        :param action: the dictionary action.
        :param state: the environment state.
        :return: action object.
        """

        # Bypass checks if action is already a PlayableAction
        if isinstance(action, (PlayableAction, TopologyAndDispatchAction)):
            return action

        assert 'action' in action

        # special treatment if action is passed as an array
        action_id = action['action']
        if isinstance(action_id, np.ndarray):
            if action_id.shape == ():
                action_id = int(action_id)
        assert isinstance(action_id, (int, np.int32, np.int, np.int64))

        # convert action id to actual action applicable to the grid
        converter_action_id = self._space_id_to_converter_id[action_id]
        action_obj = self.id_to_act.convert_act(converter_action_id)

        # automatically turn power lines on again
        if self.auto_reconnect_lines:
            action_obj = add_auto_line_reconnect_action(state=state, action=action_obj)

        return action_obj

    @override(ActionConversionInterface)
    def maze_to_space(self, action: TopologyAndDispatchAction) -> Dict[str, int]:
        """Converts environment to agent action.

        :param: action: the environment action to convert.
        :return: the dictionary action.
        """
        raise NotImplementedError

    @override(ActionConversionInterface)
    def space(self) -> spaces.Dict:
        """Returns respective gym action space.
        :return: Gym action space.
        """
        return spaces.Dict({"action": spaces.Discrete(n=self.num_active_actions)})

    @override(ActionConversionInterface)
    def noop_action(self) -> Dict[str, int]:
        """Return the noop action"""
        return {"action": 0}
