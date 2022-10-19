"""This file holds the base violation class as well the violations class."""

import dataclasses
from abc import abstractmethod
from typing import Optional, Dict, Union, List

import pandas as pd

from maze_l2rpn.grid_state_observer.grid_state_observer_types import GridElement, ViolationLevel, Contingency


@dataclasses.dataclass
class Violation:
    """A class representing a single violation on one grid element. (Usually corresponding to a contingency.)"""

    grid_element: Optional[GridElement]
    """The grid element the violation corresponds to. If this is None it corresponds to the whole grid 
    (e.g. diverging power/blackout flow violation)"""

    violation_level: ViolationLevel
    """The violation level of the violation."""

    @property
    @abstractmethod
    def numerical_violation_indicator(self) -> float:
        """Return a comparable severity indicator."""

    @abstractmethod
    def __repr__(self) -> str:
        """A string representation of the violation."""

    def __lt__(self, other: 'Violation') -> bool:
        """Compare the object with another object of the same type."""
        if self.violation_level < other.violation_level:
            return True
        elif other.violation_level < self.violation_level:
            return False

        return self.numerical_violation_indicator < other.numerical_violation_indicator

    @abstractmethod
    def __eq__(self, other: 'Violation') -> bool:
        """Compare the object with another object of the same type."""

    def as_dict(self) -> Dict[str, Union[GridElement, ViolationLevel, float, str]]:
        """Return the object as dict for table creation.

        :return: The base properties of the violation object as a dictionary.
        """
        return {'Grid Element': self.grid_element, 'Violation Level': self.violation_level.value,
                'Violation Indicator': self.numerical_violation_indicator, 'Violation Type': str(type(self).__name__)}


class Violations:
    """A class holding all violation of a current state w.r.t. to the contingencies if applicable.
    """

    def __init__(self):
        self._violations: Dict[Optional[Contingency], List[Violation]] = dict()

    def __len__(self) -> int:
        """Return the total number of violations present.

        :return: The total number of violations.
        """
        return sum(map(len, self._violations.values()))

    def set_violations(self, contingency: Optional[Contingency], list_of_violations: List[Violation]) -> None:
        """Set the list of violations for a contingency.

        :param contingency: The contingency the list of violations corresponds to. None if for state_tests.
        :param list_of_violations: The list of violations corresponding to the contingency.
        """
        self._violations[contingency] = list_of_violations

    def get_violations(self, contingency: Optional[Contingency]) -> List[Violation]:
        """Retrieve the list of violations corresponding to the contingency."""
        return self._violations[contingency]

    def get_violations_as_dict(self) -> Dict[Optional[Contingency], List[Violation]]:
        """Get all violations as a dictionary.

        :return: The violations as a dictionary.
        """
        return self._violations

    def __contains__(self, contingency: Optional[Contingency]) -> bool:
        """Checks whether the object contains the contingency."""
        return contingency in self._violations

    def count_violations(self) -> Dict[ViolationLevel, int]:
        """Count the number of violations in the given violations' dict w.r.t. the violation level.

        :return: The violations count w.r.t. the ViolationLevel.
        """
        severe_count = 0
        critical_count = 0
        normal_count = 0
        for contingency, list_of_violations in self._violations.items():
            for violation in list_of_violations:
                if violation.violation_level == ViolationLevel.severe:
                    severe_count += 1
                elif violation.violation_level == ViolationLevel.critical:
                    critical_count += 1
                elif violation.violation_level == ViolationLevel.risky:
                    normal_count += 1

        return {ViolationLevel.severe: severe_count, ViolationLevel.critical: critical_count,
                ViolationLevel.risky: normal_count}

    def _violations_as_pd_table(self) -> Optional[pd.DataFrame]:
        """Parse the violations as a panda powers dataframe.

        :return: A sorted panda powers dataframe holding all violations.
        """
        if len(self) == 0:
            return None

        rows = []
        for contingency, list_of_violations in self._violations.items():
            contingency_as_str = \
                f'{",".join([str(cc) for cc in contingency]) if contingency is not None else "CURR_STATE"}'
            if len(list_of_violations) > 0:
                rows.extend(map(lambda x: {'Contingency': contingency_as_str, **x.as_dict()}, list_of_violations))

        df = pd.DataFrame(rows)
        df = df.sort_values(by=['Violation Level', 'Violation Indicator'], ascending=[True, False])

        return df

    def get_violations_csv_table(self) -> str:
        """Retrieve the violations as a cvs string with seperator: ';'.

        :return: A string holding the violations' table as a csv.
        """

        df = self._violations_as_pd_table()
        if df is None:
            return ''

        return df.to_csv(index=False, na_rep='-', sep=';', float_format=lambda x: f"{x:.3f}")

    def __repr__(self) -> str:
        """Get a string representation of the violations.

        :return: A sorted (by severity) table of all violations.
        """
        df = self._violations_as_pd_table()
        if df is None:
            return ''

        return df.to_string(index=False, float_format=lambda x: f'{x:.3f}', na_rep='-')
