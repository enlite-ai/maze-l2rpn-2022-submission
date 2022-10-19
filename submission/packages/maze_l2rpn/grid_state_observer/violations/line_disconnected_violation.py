"""A file holding the line disconnected violation."""
import dataclasses
from typing import Dict, Union

from maze.core.annotations import override

from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel, GridElement
from maze_l2rpn.grid_state_observer.violations.violation import Violation


@dataclasses.dataclass
class LineDisconnectedViolation(Violation):
    """The line disconnected violation."""

    line_cooldown: int
    """The cooldown of the line."""

    @property
    @override(Violation)
    def numerical_violation_indicator(self) -> float:
        """Return a comparable severity indicator.

        :return: A comparable float value indicating the severity of the violation.
        """
        return self.line_cooldown

    @override(Violation)
    def __repr__(self) -> str:
        """A string representation of the violation."""
        return f'LINE_DISCONNECTED_VIOLATION: ({self.grid_element}, {self.line_cooldown}, {self.violation_level})'

    @override(Violation)
    def __eq__(self, other: 'LineDisconnectedViolation') -> bool:
        """Compare the object with another object of the same type."""
        return self.violation_level == self.violation_level and \
            self.grid_element == other.grid_element and self.line_cooldown == other.line_cooldown

    @override(Violation)
    def as_dict(self) -> Dict[str, Union[GridElement, ViolationLevel, float, str]]:
        """Return the object as dict for table creation.

        :return: The base properties of the violation object as a dictionary.
        """
        dd = super().as_dict()
        dd.update({'Line Cooldown': self.line_cooldown})
        return dd
