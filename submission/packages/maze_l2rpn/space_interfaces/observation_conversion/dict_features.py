"""Contains a state to observation interface converting the state into a vectorized feature space."""

from abc import ABC
from typing import Union, Optional, Dict

import grid2op
import gym
import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces
from maze.core.annotations import override
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.utils.factory import Factory
from omegaconf import DictConfig

from maze_l2rpn.env.grid_networkx_graph import GridNetworkxGraph
from maze_l2rpn.grid_state_observer.grid_state_observer import GridStateObserver
from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel
from maze_l2rpn.grid_state_observer.state_tests.maximum_line_load_state_test import MaximumLineLoadStateTest
from maze_l2rpn.grid_state_observer.state_tests.timesteps_line_overflown_state_test import \
    TimestepsLineOverflownStateTest
from maze_l2rpn.grid_state_observer.violations.violation import Violations


class BaseObservationConversion(ObservationConversionInterface, ABC):
    """Object representing an observation.
    For more information consider: https://grid2op.readthedocs.io/en/latest/space.html

    :param wrapped_env: The grid core environment.
    :param grid_state_observer: An optional grid state observer to decide whether a given state needs an
        agent interaction or not.
    """

    def __init__(self, wrapped_env: grid2op.Environment.Environment,
                 grid_state_observer: Optional[Union[GridStateObserver, DictConfig]]):
        self.observation_space = wrapped_env.observation_space

        self.grid_state_observer = None if grid_state_observer is None else \
            Factory(GridStateObserver).instantiate(grid_state_observer, wrapped_env=wrapped_env)

    def seed(self, seed: int) -> None:
        """Seeds the observation conversion components"""

    def is_safe_state(self, state: CompleteObservation) -> bool:
        """Check whether the current state is safe with the help of the grid state observer if given, otherwise the
        state is never safe.

        :param state: The state to be checked.
        :return: A decision if the given state is considered to be safe.
        """
        return False if self.grid_state_observer is None else self.grid_state_observer(state)

    def get_violations_as_csv(self, state: CompleteObservation) -> str:
        """Return a csv (sep=;) string representation of the violations.

        :param state: The state to be checked.
        :return: A string representation of the violations.
        """
        if self.grid_state_observer is None:
            return ''
        else:
            return self.grid_state_observer.get_violations(state).get_violations_csv_table()

    def get_violations(self, state: CompleteObservation) -> Violations:
        """Get all violations of the given state.

        :param state: The current to evaluate.
        :return: All violations as a violations object.
        """
        if self.grid_state_observer is None:
            return Violations()
        else:
            return self.grid_state_observer.get_violations(state)

    def get_total_violations_count(self, state: CompleteObservation) -> int:
        """Get the total number of violations.

        :param state: The state to be checked.
        :return: The total number of violations.
        """
        if self.grid_state_observer is None:
            return 0
        else:
            return len(self.grid_state_observer.get_violations(state))

    def update_observation_space_for_grid_state_observer(self, observation_space: spaces.Dict) -> gym.spaces.Dict:
        """Update the given observation space by adding the grid state observation space if applicable.

        :param observation_space: The current observation spaces to be updated.
        :return: The updated observation spaces with added grid state if applicable.
        """
        if self.grid_state_observer is not None:
            observation_space.spaces.update(self.grid_state_observer.gso_observation_space.spaces)
        return observation_space

    def update_observation_for_grid_state_observer(self, state: CompleteObservation,
                                                   observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Update the given observation by adding the grid state observation if applicable.

        :param state: The current state of the grid.
        :param observation: The current observation to be updated.
        :return: The updated observation with added grid state if applicable.
        """
        if self.grid_state_observer is not None:
            observation.update(self.grid_state_observer.get_observation_for_state(state))

        return observation

    @classmethod
    def _to_one_hot(cls, num_of_values: int, cur_value: int) -> np.ndarray:
        """Convert a discrete value to one_hot encoding

        :param num_of_values: The number of value this parameter can take.
        :param cur_value: The current value.
        """
        assert cur_value < num_of_values
        t = np.zeros(shape=(num_of_values,), dtype=np.float32)
        t[cur_value] = 1
        return t

    def clone_from(self, observation_conversion: 'BaseObservationConversion') -> None:
        """Reset the observation conversion to the state of the provided observation conversion.

        :param observation_conversion: The observation conversion to clone from.
        """
        if self.grid_state_observer is not None:
            self.grid_state_observer.clone_from(observation_conversion.grid_state_observer)


class ObservationConversion(BaseObservationConversion):
    """Object representing an observation.
    For more information consider: https://grid2op.readthedocs.io/en/latest/space.html

    :param grid2op_env: The grid2op environment.
    :param fix_links_for_n_sub_steps: Fix one link for substations with less than or equal the number of sub steps + 1.
    :param mask_out_storage_connections: Mask out all storage link actions.
    """

    def __init__(self,
                 grid2op_env: grid2op.Environment.Environment, fix_links_for_n_sub_steps: Optional[int],
                 mask_out_storage_connections: bool):
        grid_state_observer = GridStateObserver(
            wrapped_env=grid2op_env,
            state_tests=[MaximumLineLoadStateTest(ViolationLevel.risky, 0.98),
                         MaximumLineLoadStateTest(ViolationLevel.severe, 1.1),
                         MaximumLineLoadStateTest(ViolationLevel.critical, 1.8),
                         TimestepsLineOverflownStateTest(ViolationLevel.severe, 2),
                         TimestepsLineOverflownStateTest(ViolationLevel.critical, 3)],
            expected_outage_lookahead=2,
            post_expected_outage_state_tests=[MaximumLineLoadStateTest(ViolationLevel.risky, 0.98),
                                              MaximumLineLoadStateTest(ViolationLevel.severe, 1.1),
                                              MaximumLineLoadStateTest(ViolationLevel.critical, 1.8)],
            calculate_post_contingency_if_state_is_not_safe=False
        )

        super().__init__(grid2op_env, grid_state_observer)

        self.observation_space = grid2op_env.observation_space
        self._thermal_limit = grid2op_env.get_thermal_limit()

        self.n_gen = self.observation_space.n_gen
        self.n_load = self.observation_space.n_load
        self.n_sub = self.observation_space.n_sub
        self.n_line = self.observation_space.n_line
        self.max_buses = 2
        self.max_links = np.amax(self.observation_space.sub_info)

        self.load_to_subid = self.observation_space.load_to_subid
        self.gen_to_subid = self.observation_space.gen_to_subid
        self.line_or_to_subid = self.observation_space.line_or_to_subid
        self.line_ex_to_subid = self.observation_space.line_ex_to_subid

        # initialize topology graph
        self.link_graph = GridNetworkxGraph(space=self.observation_space,
                                            fix_links_for_n_sub_steps=fix_links_for_n_sub_steps,
                                            mask_out_storage_connections=mask_out_storage_connections)

        self._link_mask_shape = self.link_graph.n_links()
        self._link_mask_shape += 1

        self._n_features = 4 * self.n_line + self.n_sub

        self.link_masking_in_safe_state = False

    @override(ObservationConversionInterface)
    def maze_to_space(self, state: CompleteObservation) -> Dict[str, np.ndarray]:
        """Converts core environment state to space observation.
        For more information consider: https://grid2op.readthedocs.io/en/latest/observation.html#objectives

        :param state: The state returned by the powergrid env step.
        :return: The resulting dictionary observation.
        """
        is_critical_state = not self.is_safe_state(state=state)
        do_link_masking = is_critical_state or self.link_masking_in_safe_state

        # compile link mask and current adjacency
        link_mask = np.ones(self.space().spaces['link_to_set_mask'].shape, dtype=np.float32)
        if do_link_masking:
            link_mask = self.link_graph.link_mask(state)
        link_mask = np.concatenate((link_mask, np.ones(1, dtype=np.float32)))

        # current flow in powerline (n_line)
        current_flow = state.rho * self._thermal_limit

        # cumulative consumed power within substation
        sub_loads = np.zeros((state.n_sub,), dtype=np.float32)
        for load_id in range(self.observation_space.n_load):
            sub_id = self.observation_space.load_to_subid[load_id]
            sub_loads[sub_id] += state.load_p[load_id]

        # cumulative generated power within substation
        sub_gens = np.zeros((state.n_sub,), dtype=np.float32)
        for gen_id in range(self.observation_space.n_gen):
            sub_id = self.observation_space.gen_to_subid[gen_id]
            if hasattr(state, 'gen_p'):
                sub_gens[sub_id] += state.gen_p[gen_id]
            else:
                sub_gens[sub_id] += state.prod_p[gen_id]

        # power delta within substation (n_sub)
        sub_power_deltas = sub_gens - sub_loads

        # power delta between substations (n_line)
        line_deltas = np.zeros((self.n_line,), dtype=np.float32)
        for line_id in range(self.n_line):
            sub_or_id = state.line_or_to_subid[line_id]
            sub_ex_id = state.line_ex_to_subid[line_id]
            line_deltas[line_id] = sub_power_deltas[sub_or_id] - sub_power_deltas[sub_ex_id]

        features = np.concatenate([state.line_status, state.rho, current_flow, line_deltas,
                                   sub_power_deltas])

        return self.update_observation_for_grid_state_observer(observation={
            "features": features,
            "topology": state.topo_vect.astype(np.float32),
            "link_to_set_mask": link_mask
        }, state=state)

    @override(ObservationConversionInterface)
    def space_to_maze(self, observation: dict) -> CompleteObservation:
        """Converts space observation to core environment state.
        (This is most like not possible for most observation space_to_maze)
        """
        raise NotImplementedError

    @override(ObservationConversionInterface)
    def space(self) -> spaces.Dict:
        """Return the observation space shape based on the given params.

        :return: Gym space object.
        """
        float_max = np.finfo(np.float32).max
        float_min = np.finfo(np.float32).min

        return self.update_observation_space_for_grid_state_observer(gym.spaces.Dict({
            "features": spaces.Box(dtype=np.float32, shape=(self._n_features,),
                                   low=float_min, high=float_max),
            "topology": spaces.Box(dtype=np.float32, shape=(self.observation_space.dim_topo,),
                                   low=-1, high=2),
            "link_to_set_mask": spaces.Box(dtype=np.float32, shape=(self._link_mask_shape,),
                                           low=np.float32(0), high=np.float32(1))
        }))
