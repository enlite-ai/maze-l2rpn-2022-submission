
from maze.core.env.maze_env import MazeEnv
from maze.core.wrappers.wrapper import Wrapper, EnvType
from maze_l2rpn.env.maze_env import Grid2OpEnvironment


class SafeStateSkippingWrapper(Wrapper[Grid2OpEnvironment]):
    """A wrapper skipping all steps but those having observation["critical_state"]=True.

    :param env: Environment to wrap.
    """

    def __init__(self, env: MazeEnv):
        super().__init__(env)
        self._do_skip = True

    def clone_from(self, env: EnvType) -> None:
        """implementation of :class:`~maze.core.env.simulated_env_mixin.SimulatedEnvMixin`.

        Note: implementing this method is required for stateful environment wrappers.
        """
        self._do_skip = env._do_skip
        self.env.clone_from(env)

    def set_do_skip(self, do_skip: bool) -> None:
        """Set the do skip parameter of the wrapper to the given value.

        :param do_skip: Set the do skip parameter of the wrapper to this value.
        """
        self._do_skip = do_skip

    def is_do_skip(self) -> bool:
        """Check whether skipping is true.

        :return: True when skipping is currently activated.
        """
        return self._do_skip
