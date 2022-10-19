"""A file holding the diverging power flow violation."""
import dataclasses

import numpy as np

from maze.core.annotations import override

from maze_l2rpn.grid_state_observer.violations.violation import Violation


@dataclasses.dataclass
class DivergingPowerFlowViolation(Violation):
    """A diverging power flow violation."""

    @property
    @override(Violation)
    def numerical_violation_indicator(self) -> float:
        """Return a comparable severity indicator. Diverging Power flow is maximum severity."""
        return np.inf

    @override(Violation)
    def __repr__(self) -> str:
        """A string representation of the violation."""
        return f'DIVERGING_POWER_FLOW_VIOLATION'

    @override(Violation)
    def __lt__(self, other: 'Violation') -> bool:
        """Compare the object with another object of the same type."""
        return self.violation_level < other.violation_level

    @override(Violation)
    def __eq__(self, other: 'Violation') -> bool:
        """Compare the object with another object of the same type."""
        return self.violation_level == self.violation_level
