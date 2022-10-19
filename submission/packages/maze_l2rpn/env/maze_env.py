"""Contains the grid2op Maze environment."""
from copy import deepcopy
from typing import Union

from maze.core.annotations import override
from maze.core.env.action_conversion import ActionConversionInterface
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationConversionInterface
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.utils.factory import Factory, CollectionOfConfigType

from maze_l2rpn.env.core_env import Grid2OpCoreEnvironment, L2RPNSeedInfo


class Grid2OpEnvironment(MazeEnv):
    """Environment for the l2rpn challenge.

    :param core_env: Core environment or dictionary of core environment parameters.
    :param action_conversion: A dictionary with policy names as keys, containing either
                                * action to action interface implementations
                                * or a config dictionary specifying the interface instance to construct
                                  via the registration system.
    :param observation_conversion: A dictionary with policy names as keys, containing either
                                 * state to observation interface implementations
                                 * or a config dictionary specifying the interface instance to construct
                                   via the registration system.
    """

    def __init__(self,
                 core_env: Union[Grid2OpCoreEnvironment, dict],
                 action_conversion: CollectionOfConfigType,
                 observation_conversion: CollectionOfConfigType):
        core_env = Factory(Grid2OpCoreEnvironment).instantiate(core_env)

        action_conversion_dict = Factory(
            base_type=ActionConversionInterface).instantiate_collection(action_conversion,
                                                                        grid2op_env=core_env.wrapped_env)
        observation_conversion_dict = Factory(
            base_type=ObservationConversionInterface).instantiate_collection(observation_conversion,
                                                                             grid2op_env=core_env.wrapped_env)

        super().__init__(
            core_env,
            action_conversion_dict,
            observation_conversion_dict
        )
        self.is_simulated_env = False

        self.info = {}

    @override(MazeEnv)
    def seed(self, seed: Union[L2RPNSeedInfo, int]):
        """Apply action to action transformation to the action replays in the replay info."""
        if isinstance(seed, L2RPNSeedInfo):
            seed = deepcopy(seed)
            self.core_env.seed(seed)

    @override(SimulatedEnvMixin)
    def set_fast_step(self, do_fast_step: bool) -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`
        """
        if do_fast_step:
            self.observation_conversion.with_link_masking = False
        else:
            self.observation_conversion.with_link_masking = True

    @override(MazeEnv)
    def clone_from(self, env: MazeEnv) -> None:
        """Reset the maze env to the state of the provided env.

        Note, that it also clones the CoreEnv and its member variables including environment context.

        :param env: The environment to clone from.
        """
        super().clone_from(env)
        self.observation_conversion.clone_from(env.observation_conversion)
