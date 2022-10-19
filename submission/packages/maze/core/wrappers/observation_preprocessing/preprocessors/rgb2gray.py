""" Contains a rgb to gray scale conversion pre-processor. """
from typing import Tuple

import numpy as np
from gym import spaces

from maze.core.wrappers.observation_preprocessing.preprocessors.base import PreProcessor
from maze.core.annotations import override


class Rgb2GrayPreProcessor(PreProcessor):
    """An rgb-to-gray-scale conversion pre-processor.

    :param observation_space: The observation space to pre-process.
    :param rgb_dim: Dimension of the rgb channels.
    """

    def __init__(self, observation_space: spaces.Box, rgb_dim: int):
        super().__init__(observation_space)
        self.rgb_dim = rgb_dim

    @override(PreProcessor)
    def processed_shape(self) -> Tuple[int, ...]:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        new_shape = list(self._original_observation_space.shape)
        del new_shape[self.rgb_dim]
        return tuple(new_shape)

    @override(PreProcessor)
    def processed_space(self) -> spaces.Box:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        low = self.process(self._original_observation_space.low)
        high = self.process(self._original_observation_space.high)
        return spaces.Box(low=low, high=high, dtype=self._original_observation_space.dtype)

    @override(PreProcessor)
    def process(self, observation: np.ndarray) -> np.ndarray:
        """implementation of :class:`~maze.core.wrappers.observation_preprocessing.preprocessors.base.PreProcessor` interface
        """
        observation = np.swapaxes(observation, axis1=self.rgb_dim, axis2=-1)
        return np.dot(observation[..., 0:3], [0.299, 0.587, 0.114])
