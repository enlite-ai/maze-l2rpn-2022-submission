""" Implements l2rpn observation stacking as an environment wrapper. """
from typing import Dict, List

import numpy as np
from maze.core.wrappers.observation_stack_wrapper import ObservationStackWrapper


class L2RPNObservationStackWrapper(ObservationStackWrapper):
    """The observation stack wrapper with an added method for setting the observation stack."""

    def set_observation_stack(self, observation_stack: Dict[str, List[np.ndarray]]) -> None:
        """Set the observation stack of the wrapper.

        :param observation_stack: The observation stack to be set.
        """
        self._observation_stack = observation_stack

