"""Implements an RL-based power grid agent that uses a trained torch policy with predicts discrete topology change
   actions in combination with a cross entropy redispatching controller."""

import logging
import os
import time
from collections import defaultdict
from typing import Union, Dict, Any

import grid2op
import lightsim2grid
import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import CompleteObservation
from grid2op.Reward import LinesCapacityReward
from gym import spaces
from maze.core.annotations import override
from maze.core.env.maze_env import MazeEnv
from maze.core.env.structured_env import StructuredEnv
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.core.utils.config_utils import read_config, EnvFactory
from maze.core.utils.seeding import MazeSeeding
from maze.utils.bcolors import BColors
from omegaconf import DictConfig, OmegaConf

from maze_l2rpn import grid2op_utils
from maze_l2rpn.agents.maze_submission_policy import SubmissionPolicy
from maze_l2rpn.agents.recovery_module import RecoveryModule
from maze_l2rpn.env.core_env import Grid2OpCoreEnvironment
from maze_l2rpn.utils import set_logging_level, pass_action_through_wrapper_stack, \
    pass_observation_through_wrapper_stack
from .submission_params import EXPERIMENT_PATH, SubmissionParams

WRAPPER_PATH_OBS_NORM = 'maze.core.wrappers.observation_normalization.observation_normalization_wrapper.' \
                        'ObservationNormalizationWrapper'

REDISP_WRAPPER = 'maze_l2rpn.wrappers.joint_redispatching_controller_wrapper.JointRedispatchingControllerWrapper'

logger = logging.getLogger('AGENT')

other_rewards = {"LinesCapacityReward": LinesCapacityReward}


class MyStructuredLoadedAgentCritical(BaseAgent):
    """
    Maze environment based power grid agent that uses a trained torch policy (with only discrete actions) as a
    basis. Building on this a beam search is performed if a critical state leads to a blackout. Additionally, the
    presence of the CriticalStateSimulationObservation Wrapper is assumed, such that the  agent only queries the policy
    in case a critical state is encountered. Otherwise, the noop action is performed.

    :param action_space: The action space of the env.
    :param maze_env: The initialized env.
    :param policy: The policy to use for computing the actions.
    :param submission_params: The submission parameters.
    """

    def __init__(self, action_space: spaces.Space,
                 maze_env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv, Grid2OpCoreEnvironment],
                 policy: SubmissionPolicy, submission_params: SubmissionParams):
        BaseAgent.__init__(self, action_space=action_space)

        set_logging_level(logger, submission_params.LOGGING_LEVEL)

        # Environments
        self.maze_env = maze_env

        # Policy
        self.policy = policy

        # Random number generators
        self.policy_rng = np.random.RandomState()
        self.env_rng = np.random.RandomState()

        # Stats
        self.step_count = -1
        self.scenario_count = 0

        self._submission_params = submission_params
        self._skip_n_steps_after_failed_recovery = submission_params.RECOVERY_SKIP_N_STEPS_AFTER_RECOVERY
        self._only_noop = submission_params.ONLY_NOOP
        self._act_in_next_step = False

        self.recovery_module = RecoveryModule(submission_params, self.maze_env)

        self._block_recovery = False
        self.start_episode_time = time.time()

    @override(BaseAgent)
    def act(self, observation: CompleteObservation, reward: float, done: bool = False) -> PlayableAction:
        """The action that your agent will choose depending on
        the maze_state, the reward, and whether the state is terminal.

        :param observation: The maze_state passed from the (unseen) evaluation env.
        :param reward: The reward passed from the (unseen) evaluation env.
        :param done: The done passed from the (unseen) evaluation env. (This is always False).

        :return: The action to be takes next.
        """
        maze_state = observation
        self.step_count += 1

        if self._only_noop:
            return self.maze_env.action_conversion.action_space({})

        if self._block_recovery and self.step_count % self._skip_n_steps_after_failed_recovery == 0:
            self._block_recovery = False

        # Process the given maze_state --------------------------------------------------------------------------------

        # Set the maze state
        self.maze_env.set_maze_state(maze_state)
        # Convert the maze action to a space
        observation = self.maze_env.observation_conversion.maze_to_space(maze_state)
        # Recursively pass the observation through the wrapper stack
        observation = pass_observation_through_wrapper_stack(self.maze_env, observation)
        # Check if current state is critical one
        is_safe_state = 'is_safe_state' in observation and observation['is_safe_state'] == 1

        # If this not a safe state copy the maze_state properly
        if not is_safe_state or self._act_in_next_step:
            logger.debug(str(self.maze_env.observation_conversion.get_violations(maze_state)))
            maze_state = grid2op_utils.copy_maze_state(maze_state)
            self.maze_env.set_maze_state(maze_state)
            # Log information about the current state:
            lines_in_maintenance = np.where((maze_state.duration_next_maintenance > 1) &
                                            (maze_state.time_next_maintenance == 0))[0]
            lines_going_into_maintenance = np.where((0 < maze_state.time_next_maintenance) &
                                                    (maze_state.time_next_maintenance < 10))[0]
            lines_going_into_maintenance = list(zip(lines_going_into_maintenance,
                                               maze_state.time_next_maintenance[lines_going_into_maintenance]))
            offline_lines = np.where(maze_state.line_status == 0)[0]
            offline_lines = list(zip(offline_lines, maze_state.time_before_cooldown_line[offline_lines]))
            logger.info(f'[{maze_state.current_step:4}] [STATE:  {np.max(maze_state.rho):.4f}] {lines_in_maintenance}, '
                        f'{lines_going_into_maintenance}, {offline_lines}')

        # Compute the topology action to perform -----------------------------------------------------------------------
        if self.recovery_module._with_recovery and is_safe_state and not self._block_recovery \
                and not self.recovery_module.has_recovered(maze_state) \
                and self.recovery_module.recovery_possible(self.maze_env):
            action, action_is_noop = self.recovery_module.compute_recovery_action(self.maze_env, maze_state=maze_state)
            if not action_is_noop:
                logger.debug(f'[{self.step_count:4}] [{np.max(maze_state.rho):.4f}] - performing recovery action: '
                             f'(missing out on possible planed path) - {action}')
            else:
                logger.debug(f'[{self.step_count:4}] [{np.max(maze_state.rho):.4f}] - tried recoverying with no luck.. '
                             f'blocking for 10 steps: ')
                self._block_recovery = True
            self._act_in_next_step = False

        # critical state encountered, ask agent to resolve
        elif not is_safe_state or self._act_in_next_step:
            # try to resolve critical state with most probable path
            action = self.policy.compute_action(observation=observation, maze_state=maze_state, env=self.maze_env,
                                                actor_id=None, deterministic=True)
            self._act_in_next_step = self.policy.act_in_next_step()
        # all good, nothing to do
        else:
            action = self.maze_env.action_conversion.noop_action()
            self._act_in_next_step = False

        # Pass the action through the wrapper stack
        action = pass_action_through_wrapper_stack(self.maze_env, action)

        if hasattr(self.maze_env.observation_conversion, 'link_masking_in_safe_state'):
            if self._act_in_next_step:
                self.maze_env.observation_conversion.link_masking_in_safe_state = True
                self.maze_env.set_do_skip(do_skip=False)
            else:
                self.maze_env.observation_conversion.link_masking_in_safe_state = False
                self.maze_env.set_do_skip(do_skip=True)

        # Convert the action back to a playable action
        playable_action = self.maze_env.action_conversion.space_to_maze(action, maze_state)
        logger.debug(f'performing: {playable_action}')
        return playable_action

    @override(BaseAgent)
    def reset(self, obs: CompleteObservation) -> None:
        """Reset the individual components of the submission agent as well as the statistics."""
        if not self._only_noop:
            self.maze_env.set_maze_state(obs)
            self.maze_env.reset()
            self.maze_env.set_observation_stack(defaultdict(list))
            self.policy.reset(obs)

        if self.scenario_count > 0:
            logger.info(BColors.format_colored(f' Agent survived on scenario: {self.scenario_count} - '
                                               f'{self.step_count} steps, total runtime: {time.time() - self.start_episode_time:.3f}', BColors.OKBLUE))
        # Reset stats
        self.step_count = -1
        self.scenario_count += 1
        self._block_recovery = False
        self.start_episode_time = time.time()
        self._act_in_next_step = False

    @override(BaseAgent)
    def seed(self, seed: int) -> None:
        """The seeding method of the base agent

        :param seed: The seed to set.
        """
        # Generate an agent seed and set the seed globally for the model initialization
        logger.debug(f'seeding policy with: {seed}')
        self.maze_env.seed(seed)
        self.policy_rng = np.random.RandomState(seed)
        self.policy.seed(MazeSeeding.generate_seed_from_random_state(self.policy_rng))

    @classmethod
    def build(cls, grid2op_env: grid2op.Environment.Environment,
              this_directory_path: str, experiment_path: str,
              submission_params: 'SubmissionParams') -> 'MyStructuredLoadedAgentCritical':
        """Factory method for creating the MyStructuredLoadedAgentCritical class object.

        :param grid2op_env: The initialized grid2op environment.
        :param this_directory_path: The current directory path.
        :param experiment_path: The experiment path.
        :return: The Initialized agent to be used for evaluation.
        """

        # Retrieve experiment files
        hydra_config_path = os.path.join(this_directory_path, experiment_path, '.hydra/config.yaml')

        # Parse Hydra config file
        hydra_config_unresolved = DictConfig(read_config(hydra_config_path))
        hydra_config: Dict[str, Any] = OmegaConf.to_container(hydra_config_unresolved, resolve=True)

        # Update the observation normalization wrapper path for statistics
        assert WRAPPER_PATH_OBS_NORM in hydra_config['wrappers']
        hydra_config['wrappers'][WRAPPER_PATH_OBS_NORM]['statistics_dump'] = \
            os.path.join(this_directory_path, experiment_path,
                         hydra_config['wrappers'][WRAPPER_PATH_OBS_NORM]['statistics_dump'])

        # Update the action selection vector path for unitary actions
        if 'action_selection_vector_dump' in hydra_config['env']['action_conversion'][0]:
            dump_file = hydra_config['env']['action_conversion'][0]['action_selection_vector_dump']
            hydra_config['env']['action_conversion'][0]['action_selection_vector_dump'] = \
                os.path.join(this_directory_path, experiment_path, dump_file)

        # Update path to CA redispatching
        input_dir = hydra_config['wrappers'][REDISP_WRAPPER]['contingency_redispatching_controller']['input_dir']
        hydra_config['wrappers'][REDISP_WRAPPER]['contingency_redispatching_controller']['input_dir'] = \
            os.path.join(this_directory_path, experiment_path, input_dir)

        # substitute given environment
        hydra_config['env']['core_env']['power_grid'] = grid2op_env

        logger.warning(f'Building agent for competition: wcci_2022 - with grid2op version: {grid2op.__version__} '
                       f'and lightsim2grid version: {lightsim2grid.__version__}')

        # Build Maze-Environment
        maze_env = EnvFactory(hydra_config['env'], hydra_config['wrappers'])()

        policy = submission_params.maze_policy.build(hydra_config=hydra_config, maze_env=maze_env,
                                                     submission_params=submission_params,
                                                     this_directory_path=this_directory_path,
                                                     experiment_path=experiment_path)

        # Initialize Submission Agent
        return cls(action_space=maze_env.action_space, maze_env=maze_env,
                   policy=policy, submission_params=submission_params)


def make_agent(grid2op_env: grid2op.Environment.Environment, this_directory_path: str) -> BaseAgent:
    """Build the desired agent for the evaluation

    :param grid2op_env: The initialized grid2op environment.
    :param this_directory_path: The current directory path.
    :return: The Initialized agent to be used for evaluation.
    """
    return MyStructuredLoadedAgentCritical.build(grid2op_env, this_directory_path, EXPERIMENT_PATH,
                                                 submission_params=SubmissionParams())
