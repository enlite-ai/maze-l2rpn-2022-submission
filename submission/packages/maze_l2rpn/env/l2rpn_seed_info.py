from dataclasses import dataclass
from typing import Tuple, Any


@dataclass
class L2RPNSeedInfo:
    """Represents seed information of the Grid2Op env."""

    env_index: int
    """Index of the environment within the multi-environment."""

    chronic_id: int
    """Index of the Grid2Op chronics data."""

    random_seed: int
    """Seed for the pseudo-random generators."""

    fast_forward: int
    """Applied to Grid2Op fast_forward_chronics() method."""

    actions: Tuple[Any, ...]
    """Sequence of actions applied on env reset()."""

    def __eq__(self, other: 'L2RPNSeedInfo'):
        return (self.env_index == other.env_index and
                self.chronic_id == other.chronic_id and
                self.random_seed == other.random_seed and
                self.fast_forward == other.fast_forward and
                self.actions == other.actions)

    def __hash__(self):
        return hash(tuple([self.env_index, self.chronic_id, self.random_seed, self.fast_forward, tuple(self.actions)]))
