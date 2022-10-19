"""Interface for redispatching controllers that act on top of state and can be used
in conjunction with topology policies."""

from abc import abstractmethod
from typing import Union, Tuple, Dict, Any, Optional, List

from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation

from maze_l2rpn.env.l2rpn_seed_info import L2RPNSeedInfo


class RedispatchingController:
    """Interface for redispatching controllers that act on top of state and can be used
    in conjunction with topology policies."""

    @abstractmethod
    def compute_action(self,
                       state: CompleteObservation,
                       line_contingencies: List[int],
                       lines_to_relieve: List[int],
                       joint_action: Optional[PlayableAction] = None) -> Tuple[PlayableAction, Dict[str, Any]]:
        """Compute the redispatching action (redispatching / curtailment / storages)
        based on the current state.

        :return: Tuple of (playable grid2op action, info dict).
        """

    @abstractmethod
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seeds the random generator of the controller."""
