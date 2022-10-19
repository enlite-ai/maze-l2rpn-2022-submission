"""A file holding the line load violation."""
import dataclasses
from typing import Dict, Union

import numpy as np

from maze.core.annotations import override

from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel, GridElement
from maze_l2rpn.grid_state_observer.violations.violation import Violation


@dataclasses.dataclass
class LineLoadViolation(Violation):
    """A line load threshold violation."""

    line_load: float
    """The load of the line exceeding the threshold."""

    line_threshold: float
    """The threshold of the line (that has been exceeded)."""

    @property
    @override(Violation)
    def numerical_violation_indicator(self) -> float:
        """Return a comparable severity indicator.

        :return: A comparable float value indicating the severity of the violation.
        """
        assert self.line_load >= self.line_threshold
        return self.line_load / self.line_threshold

    @property
    def relative_overload(self) -> float:
        """Get the relative overload of the line."""
        return self.line_load - self.line_threshold

    @override(Violation)
    def __repr__(self) -> str:
        """A string representation of the violation."""
        return f'LINE_LIMIT_VIOLATION: ({self.grid_element}, {self.numerical_violation_indicator:.4f}, ' \
               f'{self.violation_level})'

    @override(Violation)
    def __lt__(self, other: 'LineLoadViolation') -> bool:
        """Compare the object with another object of the same type."""
        if self.violation_level < other.violation_level:
            return True
        elif other.violation_level < self.violation_level:
            return False

        return self.numerical_violation_indicator < other.numerical_violation_indicator

    @override(Violation)
    def __eq__(self, other: 'LineLoadViolation') -> bool:
        """Compare the object with another object of the same type."""
        return self.violation_level == self.violation_level and np.isclose(self.line_load, other.line_load) and \
            self.grid_element == other.grid_element and self.line_threshold == other.line_threshold

    @override(Violation)
    def as_dict(self) -> Dict[str, Union[GridElement, ViolationLevel, float, str]]:
        """Return the object as dict for table creation.

        :return: The base properties of the violation object as a dictionary.
        """
        dd = super().as_dict()
        dd.update({'Line Load': self.line_load, 'Line Threshold': self.line_threshold})
        return dd
