""" Import normalization strategies to enable import shortcuts. """
from maze.core.wrappers.observation_normalization.normalization_strategies.base import ObservationNormalizationStrategy
from maze.core.wrappers.observation_normalization.normalization_strategies.mean_zero_std_one import \
    MeanZeroStdOneObservationNormalizationStrategy
from maze.core.wrappers.observation_normalization.normalization_strategies.range_zero_one import \
    RangeZeroOneObservationNormalizationStrategy

assert issubclass(RangeZeroOneObservationNormalizationStrategy, ObservationNormalizationStrategy)
assert issubclass(MeanZeroStdOneObservationNormalizationStrategy, ObservationNormalizationStrategy)