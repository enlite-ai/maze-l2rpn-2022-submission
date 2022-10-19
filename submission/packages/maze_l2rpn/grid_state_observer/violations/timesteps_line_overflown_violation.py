"""A file holding the violation of lines that have been overflown for specified number of timesteps."""
import dataclasses
from typing import Dict, Union

import numpy as np

from maze.core.annotations import override

from maze_l2rpn.grid_state_observer.grid_state_observer_types import GridElement, ViolationLevel
from maze_l2rpn.grid_state_observer.violations.violation import Violation


@dataclasses.dataclass
class TimestepsLineOverflownViolation(Violation):
    """A violation for lines that have been overflown for specified number of timesteps."""

    line_load: float
    """The load of the line exceeding the threshold."""

    timesteps_overflown: int
    """The threshold of the line (that has been exceeded)."""

    @property
    @override(Violation)
    def numerical_violation_indicator(self) -> float:
        """Return a comparable severity indicator.

        :return: A comparable float value indicating the severity of the violation.
        """
        return self.timesteps_overflown

    @override(Violation)
    def __repr__(self) -> str:
        """A string representation of the violation."""
        return f'TIMESTEPS_LINE_OVERFLOWN_VIOLATION: ({self.grid_element}, {self.numerical_violation_indicator:.4f}, ' \
               f'{self.violation_level})'

    @override(Violation)
    def __eq__(self, other: 'TimestepsLineOverflownViolation') -> bool:
        """Compare the object with another object of the same type."""
        return self.violation_level == self.violation_level and np.isclose(self.line_load, other.line_load) and \
            self.grid_element == other.grid_element and self.timesteps_overflown == other.timesteps_overflown

    @override(Violation)
    def as_dict(self) -> Dict[str, Union[GridElement, ViolationLevel, float, str]]:
        """Return the object as dict for table creation.

        :return: The base properties of the violation object as a dictionary.
        """
        dd = super().as_dict()
        dd.update({'Line Load': self.line_load, 'Timesteps Line Overflown': self.timesteps_overflown})
        return dd
