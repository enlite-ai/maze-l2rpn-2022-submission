"""A state test that check whether a line has been overflown for a specified number of time steps."""
from typing import List, Union

import numpy as np
from grid2op.Observation import CompleteObservation
from maze.core.annotations import override

from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel, GridElement, GridElementType
from maze_l2rpn.grid_state_observer.state_tests.state_test_interface import StateTestInterface
from maze_l2rpn.grid_state_observer.violations.timesteps_line_overflown_violation import TimestepsLineOverflownViolation
from maze_l2rpn.grid_state_observer.violations.violation import Violation


class TimestepsLineOverflownStateTest(StateTestInterface):
    """A state test that check whether has been overflown for a specified number of time steps.

    :param violation_level: The violation level the state tests result should correspond to.
    :param max_timesteps_overflown: The number of time steps to check whether a line has been overloaded for.
    """

    def __init__(self, violation_level: Union[str, ViolationLevel], max_timesteps_overflown: int):
        super().__init__(violation_level)
        self._max_timesteps_overflown = max_timesteps_overflown

    @override(StateTestInterface)
    def __call__(self, state: CompleteObservation) -> List[Violation]:
        """Return the number of power-lines that have been overloaded for specified number of time steps.

        :param state: The state to be checked.
        :return: A list of all violations, that is a list of all time step violations occurring.
        """
        violations = list()
        for i in np.where(state.timestep_overflow >= self._max_timesteps_overflown)[0]:
            violations.append(TimestepsLineOverflownViolation(line_load=state.rho[i],
                                                              grid_element=GridElement(GridElementType.line, int(i)),
                                                              violation_level=self._violation_level,
                                                              timesteps_overflown=self._max_timesteps_overflown))

        return violations

    @override(StateTestInterface)
    def clone_from(self, state_test: 'TimestepsLineOverflownStateTest') -> None:
        """Reset the state test to the state of the provided state test.

        :param state_test: The state test to clone from.
        """
        self._max_timesteps_overflown = state_test._max_timesteps_overflown
