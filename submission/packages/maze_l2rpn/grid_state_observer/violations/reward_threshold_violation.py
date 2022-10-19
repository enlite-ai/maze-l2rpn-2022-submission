"""A file holding the line load violation."""
import dataclasses
from typing import Dict, Union

import numpy as np

from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel, GridElement
from maze_l2rpn.grid_state_observer.violations.violation import Violation
from maze.core.annotations import override


@dataclasses.dataclass
class RewardThresholdViolation(Violation):
    """A line load threshold violation."""

    state_reward: float

    reward_threshold: float

    reward_name: str

    @property
    @override(Violation)
    def numerical_violation_indicator(self) -> float:
        """Return a comparable severity indicator.

        :return: A comparable float value indicating the severity of the violation.
        """
        assert self.state_reward >= self.state_reward
        return self.state_reward / self.reward_threshold

    @property
    def relative_overload(self) -> float:
        """Get the relative overload of the line."""
        return self.state_reward - self.reward_threshold

    @override(Violation)
    def __repr__(self) -> str:
        """A string representation of the violation."""
        return f'REWARD_THRESHOLD_VIOLATION: ({self.reward_name}, {self.numerical_violation_indicator:.4f}, ' \
               f'{self.state_reward:.3f}, ' \
               f'{self.violation_level})'

    @override(Violation)
    def __lt__(self, other: 'RewardThresholdViolation') -> bool:
        """Compare the object with another object of the same type."""
        if self.violation_level < other.violation_level:
            return True
        elif other.violation_level < self.violation_level:
            return False

        return self.numerical_violation_indicator < other.numerical_violation_indicator

    @override(Violation)
    def __eq__(self, other: 'RewardThresholdViolation') -> bool:
        """Compare the object with another object of the same type."""
        return self.violation_level == self.violation_level and np.isclose(self.state_reward, other.state_reward) and \
            self.grid_element == other.grid_element and self.reward_threshold == other.reward_threshold

    @override(Violation)
    def as_dict(self) -> Dict[str, Union[GridElement, ViolationLevel, float, str]]:
        """Return the object as dict for table creation.

        :return: The base properties of the violation object as a dictionary.
        """
        dd = super().as_dict()
        dd.update({'Reward Name': self.reward_name, 'Reward Threshold': self.reward_threshold, 'Reward Value': self.state_reward})
        return dd
