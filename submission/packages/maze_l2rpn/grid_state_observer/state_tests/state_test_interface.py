"""Holds the interface for state tests. These test should validate the current state of the grid and return an
integer value indicated how many violations are present in the grid."""
from abc import abstractmethod
from typing import List, Union

from grid2op.Observation import CompleteObservation

from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel
from maze_l2rpn.grid_state_observer.violations.violation import Violation


class StateTestInterface:
    """Holds the interface for state tests. These test should validate the current state of the grid and return an
    integer value indicated how many violations are present in the grid.

    :param violation_level: The violation level the state tests result should correspond to.
    """

    def __init__(self, violation_level: Union[str, ViolationLevel]):
        self._violation_level = ViolationLevel.build(violation_level)

    @abstractmethod
    def __call__(self, state: CompleteObservation) -> List[Violation]:
        """Test the current grid state for violations.

        :param state: The current grid state.
        :return: A list of all violations occurring.
        """

    @abstractmethod
    def clone_from(self, state_test: 'StateTestInterface') -> None:
        """Reset the state test to the state of the provided state test.

        :param state_test: The state test to clone from.
        """
