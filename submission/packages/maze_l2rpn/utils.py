"""Contains utility methods for the l2rpn repo"""

import logging
from typing import List, Optional, Callable, Union, Tuple

import numpy as np
from maze.core.env.action_conversion import ActionType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.wrappers.wrapper import ActionWrapper, ObservationWrapper, Wrapper
from omegaconf import DictConfig

from maze.core.env.maze_env import MazeEnv
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.utils.factory import Factory, ConfigType
from maze.core.utils.seeding import MazeSeeding
from maze.core.wrappers.observation_normalization.observation_normalization_wrapper import \
    ObservationNormalizationWrapper

from maze_l2rpn.env.maze_env import Grid2OpEnvironment
from maze_l2rpn.wrappers.l2rpn_observation_stack_wrapper import L2RPNObservationStackWrapper
from maze_l2rpn.wrappers.safe_sate_skipping_wrapper import SafeStateSkippingWrapper


def prepare_simulated_env(exclude_wrappers: Optional[List[str]], main_env: MazeEnv, policy_rng: np.random.RandomState,
                          simulated_env: Union[SimulatedEnvMixin, Callable[[], SimulatedEnvMixin], ConfigType]) -> \
        Union[SimulatedEnvMixin, Callable[[], SimulatedEnvMixin], ConfigType]:
    """
    Prepares a simulated environment by excluding certain wrappers, instantiating the env and setting normalization
    statistics.

    :param exclude_wrappers: Wrappers to exclude from simulated environment to reduce overhead.
    :param main_env: The main environment object for copying stats.
    :param policy_rng: A numpy RandomState
    :param simulated_env: A model environment instance used for sampling.
    :return: Instantiated and prepared simulated_env
    """
    # instantiate simulated env from config
    # potentially exclude wrappers from simulated env to be instantiated
    if exclude_wrappers:
        wrapper_config = DictConfig(simulated_env['wrappers'].__dict__['_content'])
        for wrapper in exclude_wrappers:
            if wrapper in wrapper_config:
                logging.info(f"Excluding '{wrapper}' from simulated environment!")
                wrapper_config.pop(wrapper)
        simulated_env['wrappers'] = wrapper_config

    # instantiate env
    simulated_env = Factory(base_type=SimulatedEnvMixin).instantiate(simulated_env)
    simulated_env.seed(MazeSeeding.generate_seed_from_random_state(policy_rng))

    # set normalization statistics
    if isinstance(simulated_env, ObservationNormalizationWrapper):
        assert isinstance(main_env, ObservationNormalizationWrapper)
        simulated_env.set_normalization_statistics(main_env.get_statistics())

    return simulated_env


def set_logging_level(logger: logging.Logger, logging_level: str) -> None:
    """Set the logging level of the given logger.

    :param logger: The logger to set the level of.
    :param logging_level: The logging_level to set.
    """
    assert logging_level in ['debug', 'debug_and_plot', 'info', 'warning']
    if logging_level in ['debug', 'debug_and_plot']:
        logger.setLevel(logging.DEBUG)
    elif logging_level == 'info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


def pass_action_through_wrapper_stack(env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv],
                                      maze_action: ActionType) -> ActionType:
    """Get the action by recursively passing it thorough the wrapper stack. (Happens implicitly at training time.)

    :param env: The current env in the wrapper stack.
    :param maze_action: The current action in the wrapper stack.

    :return: The maze action passed through the complete wrapper stack.
    """

    if type(env) == Grid2OpEnvironment:
        return maze_action
    elif type(env).__bases__[0] == ActionWrapper:
        maze_action = env.action(maze_action)
        return pass_action_through_wrapper_stack(env.env, maze_action)
    else:
        return pass_action_through_wrapper_stack(env.env, maze_action)


def pass_observation_through_wrapper_stack(env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv],
                                           observation: ObservationType) -> ObservationType:
    """Get the observation by recursively passing it thorough the wrapper stack. (Happens implicitly at training time.)

    :param env: The current env in the wrapper stack.
    :param observation: The current observation in the wrapper stack.

    :return: The maze observation passed through the complete wrapper stack.
    """

    observation, _ = pass_observation_through_wrapper_stack_rec(env, observation)
    return observation


def pass_observation_through_wrapper_stack_rec(env: MazeEnv, observation: ObservationType) -> Tuple[
    ObservationType, bool]:
    """Get the observation by recursively passing it thorough the wrapper stack.

    :param env: The current env in the wrapper stack.
    :param observation: The current observation in the wrapper stack.

    :return A tuple holding the processed maze-observation and a boolean indicating whether to stop the conversion.
    """
    if type(env).__bases__[0] == ObservationWrapper or type(env) == L2RPNObservationStackWrapper:
        obs, stop_conversion = pass_observation_through_wrapper_stack_rec(env.env, observation)
        if not stop_conversion:
            obs = env.observation(obs)
        return obs, stop_conversion
    elif type(env).__bases__[0] in (Wrapper, ActionWrapper):
        obs, stop_conversion = pass_observation_through_wrapper_stack_rec(env.env, observation)
        if type(env) == SafeStateSkippingWrapper and obs['is_safe_state'] == 1 and env.is_do_skip():
            # If critical state is not set, break wrapper stack, since it performs unnecessary computation
            stop_conversion = True
        return obs, stop_conversion
    else:
        return observation, False
