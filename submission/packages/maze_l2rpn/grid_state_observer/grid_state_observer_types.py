"""A file holding all type definitions for the grid state observer."""
from enum import Enum
from typing import Tuple, Union, Dict

from grid2op.Observation.completeObservation import CompleteObservation


class StringEnum(Enum):
    """A helper class for enums holding only strings, with one method for building them."""

    @classmethod
    def build(cls, value: Union[str, 'StringEnum']) -> 'StringEnum':
        """Load the class instance from str if necessary.

        :param value: The value as the initialized enum or as a string.
        :return: The initialized enum instance.
        """
        if isinstance(value, cls):
            return value

        assert value in [e.value for e in cls], \
            f'Unsupported grid element type given: {value} --> Supported values: ({[e.value for e in cls]})'
        return {e.value: e for e in cls}[value]


class ViolationLevel(StringEnum):
    """The violation levels used to classify contingency violations.
    """
    severe = 'SEVERE'
    """A state where several elements such as lines and transformers become overloaded and risk damage."""
    critical = 'CRITICAL'
    """A state when the power system becomes unstable and will quickly collapse"""
    risky = 'RISKY'
    """A state that is not a safe but also does not violate any severe or critical contingencies."""

    def __lt__(self, other: 'ViolationLevel'):
        """Compare the object with another object of the same type."""
        if self.severe and other.critical:
            return True
        elif self.risky and (other.critical or other.severe):
            return True
        else:
            return False

    def __eq__(self, other: Union['ViolationLevel', str]):
        """Compare the object with another object of the same type."""
        if isinstance(other, str):
            return self.value == other

        return self.value == other.value

    def __hash__(self):
        """Return an int value for the elements of the enum."""
        return self.value.__hash__()


class GridElementType(StringEnum):
    """An enum holding the different grid element types."""

    line = 'LINE'
    generator = 'GENERATOR'
    load = 'LOAD'


class GridElement:
    """An object representing a single grid element.

    :param element_type: The type of the grid element, as a string or enum.
    :param idx: The idx of the grid element.
    """

    def __init__(self, element_type: Union[GridElementType, str], idx: int):
        self.element_type = GridElementType.build(element_type)
        self.idx = idx

    def __repr__(self) -> str:
        """Get a string representation of the object instance."""
        return f'({self.element_type.value}, {self.idx})'

    def __eq__(self, other: 'GridElement') -> bool:
        """Check whether the other object is equal to this one."""
        return self.element_type.value == other.element_type.value and self.idx == other.idx

    def __hash__(self) -> int:
        """Get a hash of the object."""
        return tuple([self.element_type.value, self.idx]).__hash__()

    @classmethod
    def build(cls, value: Union['GridElement', Tuple[str, int]]) -> 'GridElement':
        """Build a grid element instance form a tuple if necessary.

        :param value: The value used to build the instance. Either a tuple for an already instantiated object.
        """
        if isinstance(value, cls):
            return value

        assert isinstance(value[0], (GridElementType, str))
        assert isinstance(value[1], int)
        return cls(value[0], value[1])


class Contingency(tuple):
    """The base type of a contingency is a tuple of GridElements"""

    def to_action_dict(self, state: CompleteObservation) -> Dict:
        """Construct a grid2op action simulating this contingency."""
        action = dict()
        for grid_elem in self:
            if grid_elem.element_type == GridElementType.line:
                if not state.line_status[grid_elem.idx]:
                    continue
                if 'set_line_status' not in action:
                    action['set_line_status'] = list()
                action["set_line_status"].append((grid_elem.idx, -1))
            else:
                raise NotImplementedError(
                    f'Outages for grid elements of type: {grid_elem.element_type} can not be simulated at '
                    f'this point.')

        return action


class ExpectedContingency(Contingency):
    """The expected contingency is a tuple of GridElements. This is used to distinguish maintenance contingencies for
    normal ones."""
    pass
