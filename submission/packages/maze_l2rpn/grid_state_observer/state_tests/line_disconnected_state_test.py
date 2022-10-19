"""Critical state observer that also check whether any line has been disconnected."""
from typing import List, Union

import numpy as np
from grid2op.Observation import CompleteObservation

from maze.core.annotations import override

from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel, GridElement, GridElementType
from maze_l2rpn.grid_state_observer.state_tests.state_test_interface import StateTestInterface
from maze_l2rpn.grid_state_observer.violations.line_disconnected_violation import LineDisconnectedViolation
from maze_l2rpn.grid_state_observer.violations.violation import Violation


class LineDisconnectedStateTest(StateTestInterface):
    """A state test that checks if a line has been disconnected by the backend.

    :param violation_level: The violation level the state tests result should correspond to.
    """

    def __init__(self, violation_level: Union[str, ViolationLevel]):
        super().__init__(violation_level)

    @override(StateTestInterface)
    def __call__(self, state: CompleteObservation) -> List[Violation]:
        """Return a LineDisconnectedViolation for each line that has been disconnected.

        :param state: The state to be checked.
        :return: A list of all violations, that is a list of all line limit violations occurring.
        """
        violations = list()
        for i in np.where(~state.line_status &
                          (state.time_before_cooldown_line == state._obs_env.parameters.NB_TIMESTEP_RECONNECTION))[0]:
            violations.append(LineDisconnectedViolation(grid_element=GridElement(GridElementType.line, int(i)),
                                                        violation_level=self._violation_level,
                                                        line_cooldown=state.time_before_cooldown_line[i]))
        return violations

    @override(StateTestInterface)
    def clone_from(self, state_test: 'LineDisconnectedStateTest') -> None:
        """Reset the state test to the state of the provided state test.

        :param state_test: The state test to clone from.
        """
