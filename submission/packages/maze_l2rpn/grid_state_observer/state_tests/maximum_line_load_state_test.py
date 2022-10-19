"""Critical state observer that also check the max-rho-delta to the last observation"""
from typing import List, Union

import numpy as np
from grid2op.Observation import CompleteObservation

from maze.core.annotations import override

from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel, GridElement, GridElementType
from maze_l2rpn.grid_state_observer.state_tests.state_test_interface import StateTestInterface
from maze_l2rpn.grid_state_observer.violations.line_load_violation import LineLoadViolation
from maze_l2rpn.grid_state_observer.violations.violation import Violation


class MaximumLineLoadStateTest(StateTestInterface):
    """A state test that focuses solely on the maximal rho of the given state.

    :param violation_level: The violation level the state tests result should correspond to.
    :param max_line_load: The maximum line capacity be for considered a CRITICAL state.
    """

    def __init__(self, violation_level: Union[str, ViolationLevel], max_line_load: float):
        super().__init__(violation_level)
        self._max_line_load = max_line_load

    @override(StateTestInterface)
    def __call__(self, state: CompleteObservation) -> List[Violation]:
        """Return the number of power-lines with a rho greater than the defined max_rho.

        :param state: The state to be checked.
        :return: A list of all violations, that is a list of all line limit violations occurring.
        """
        violations = list()
        for i in np.where((state.rho > self._max_line_load) + np.isnan(state.rho) > 0)[0]:
            violations.append(LineLoadViolation(line_load=state.rho[i], line_threshold=self._max_line_load,
                                                grid_element=GridElement(GridElementType.line, int(i)),
                                                violation_level=self._violation_level))
        return violations

    @override(StateTestInterface)
    def clone_from(self, state_test: 'MaximumLineLoadStateTest') -> None:
        """Reset the state test to the state of the provided state test.

        :param state_test: The state test to clone from.
        """
        self._max_line_load = state_test._max_line_load
