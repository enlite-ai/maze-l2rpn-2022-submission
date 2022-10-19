"""Contains a simple simulation search policy where the search is guided by a trained torch policy."""
import logging
import os
import time
from typing import Union, Tuple, Optional, Sequence

import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from maze.core.agent.policy import Policy
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.env.maze_env import MazeEnv
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import StructuredEnv, ActorID
from maze.core.env.structured_env_spaces_mixin import StructuredEnvSpacesMixin
from maze.utils.bcolors import BColors
from omegaconf import DictConfig

from maze_l2rpn.agents.maze_submission_policy import SubmissionPolicy
from maze_l2rpn.env.core_env import Grid2OpCoreEnvironment
from maze_l2rpn.maze_extensions.serialized_torch_policy_p_36 import SerializedTorchPolicyP36
from maze_l2rpn.simulate.grid2op_simulate import grid2op_simulate
from maze_l2rpn.utils import prepare_simulated_env, set_logging_level
from maze_l2rpn.wrappers.joint_redispatching_controller_wrapper import JointRedispatchingControllerWrapper

WRAPPER_PATH_OBS_NORM = 'maze.core.wrappers.observation_normalization.observation_normalization_wrapper.' \
                        'ObservationNormalizationWrapper'

logger = logging.getLogger('SIM_SEARCH ')


def is_executable(maze_action: PlayableAction, maze_state: CompleteObservation) -> bool:
    """Check if an action can be executed.

    :param maze_action: The action to check.
    :param maze_state: The current env state.
    :return: True if action can be executed; else False.
    """
    # skip action where substations are in cooldown
    topological_impact = maze_action.get_topological_impact(maze_state.line_status)
    impacted_substations = np.where(topological_impact[1])[0]
    if np.any(maze_state.time_before_cooldown_sub[impacted_substations] > 0):
        return False

    # skip actions where lines are in cooldown
    impacted_power_lines = np.where(topological_impact[0])[0]
    if np.any(maze_state.time_before_cooldown_line[impacted_power_lines] > 0):
        return False

    return True


class SimulationSearchSubmissionAgent(SubmissionPolicy):
    """Maze environment based power grid agent that uses a trained torch policy (with only discrete actions) as a
    basis to search for viable congestion management actions utilizing the grid simulation.

    :param simulated_env: Configuration to instantiate a helper environment for action conversion.
    :param structured_policy: The torch policy to guide the simulation-based search.
    :param submission_params: Additional parametrization of the agent (see submission_params.py for details).
    """

    def __init__(self,
                 simulated_env: Union[StructuredEnv, StructuredEnvSpacesMixin, MazeEnv, Grid2OpCoreEnvironment],
                 structured_policy: TorchPolicy,
                 submission_params: 'SubmissionParams'):
        super().__init__()

        set_logging_level(logger, submission_params.LOGGING_LEVEL)

        self.action_conversion = simulated_env.action_conversion

        self.torch_policy = structured_policy

        self.rng = np.random.RandomState()

        self._n_candidates = submission_params.N_CANDIDATES
        self._n_candidates_unsafe = submission_params.N_CANDIDATES_UNSAFE
        self._min_candidates = submission_params.MIN_CANDIDATES
        self._early_stop_max_rho = submission_params.EARLY_STOPPING_MAX_RHO

    @override(Policy)
    def compute_action(self,
                       observation: ObservationType,
                       maze_state: Optional[CompleteObservation],
                       env: Optional[MazeEnv],
                       actor_id: Optional[ActorID] = None,
                       deterministic: bool = False) -> ActionType:
        """Implementation of :py:attr:`~maze.core.agent.policy.Policy.compute_action`.

        :param observation: Current observation of the environment
        :param maze_state: Current state representation of the environment
                           (only provided if `needs_state()` returns True)
        :param env: The environment instance (only provided if `needs_env()` returns True)
        :param actor_id: ID of the actor to query policy for
                         (does not have to be provided if there is only one actor and one policy in this environment)
        :param deterministic: Specify if the action should be computed deterministically
        :return: Next action to take
        """

        # record time required for action computation
        start_time = time.time()

        # compute top action candidates
        actions, scores = self.compute_top_action_candidates(observation,
                                                             num_candidates=self._n_candidates_unsafe,
                                                             maze_state=maze_state,
                                                             env=env,
                                                             actor_id=actor_id)

        # simulate risk of top candidates
        best_action, lowest_risk, best_index = None, np.inf, None
        n_simulations = 0
        if isinstance(env, JointRedispatchingControllerWrapper):
            env.last_top_rhos = []
            env.last_top_maze_actions = []
        for i, action in enumerate(actions):
            # convert agent action to grid2op action
            maze_action = self.action_conversion.space_to_maze(action, state=maze_state)

            # skip action where substations are in cooldown
            if not is_executable(maze_action, maze_state):
                continue

            # simulation next step with action candidate
            obs_simulate, reward_simulate, done_simulate, info_simulate = \
                grid2op_simulate(maze_state, maze_action, use_forcast=True, post_state_preparation=False,
                                 assert_env_dynamics=False)
            n_simulations += 1

            # estimate risk of resulting state and keep best one
            risk = self._risk(state=obs_simulate, done=done_simulate)
            if risk < lowest_risk:
                lowest_risk = risk
                best_action = action
                best_index = i

            # preserve top actions
            if isinstance(env, JointRedispatchingControllerWrapper):
                env.append_top_actions(risk=risk, maze_action=maze_action)

            if self._early_stop_max_rho is not None and lowest_risk < self._early_stop_max_rho \
                    and (i + 1) >= self._min_candidates:
                break

            if lowest_risk < 1.0 and i > self._n_candidates:
                break

            # even in cases where the risk is np.inf, we need to take a valid action
            if best_action is None:
                best_action = action

        # compute overall time
        total_time = time.time() - start_time

        # some logging
        color = BColors.OKGREEN if n_simulations < self._n_candidates else BColors.WARNING
        if lowest_risk >= 1.0:
            color = BColors.FAIL
        logger.info(BColors.format_colored(
            f'[{maze_state.current_step:4}] Action (idx={best_index}) computed '
            f'with max rho {lowest_risk:.3f} after {n_simulations} simulations in {total_time:.2f}s.',
            color=color))

        return best_action

    @override(Policy)
    def compute_top_action_candidates(self,
                                      observation: ObservationType,
                                      num_candidates: int,
                                      maze_state: Optional[CompleteObservation],
                                      env: Optional[MazeEnv],
                                      actor_id: Union[str, int] = None) -> Tuple[Sequence[ActionType], Sequence[float]]:
        """
        Implementation of :py:attr:`~maze.core.agent.policy.Policy.compute_top_action_candidates`.
        """

        # just forward call to internal torch policy
        actions, probs = self.torch_policy.compute_top_action_candidates(
            observation=observation,
            num_candidates=num_candidates,
            maze_state=maze_state,
            env=env,
            actor_id=actor_id
        )
        return actions, probs

    @classmethod
    def _risk(cls, state: CompleteObservation, done: bool) -> float:
        """Risk function for a given state.

        :param state: The state to compute the risk value for.
        :return: Risk of provided state.
        """
        if done:
            return np.inf

        if np.all(state.rho < 1.0):
            risk = state.rho.max()
        else:
            risk = np.sum(state.rho[state.rho >= 1.0])

        return risk

    @override(SubmissionPolicy)
    def act_in_next_step(self) -> bool:
        """If a path is created to the best score let the agent act in the next state event if it is a safe state.

        :return: A bool indicating whether the agent should act in the next state.
        """
        return False

    @override(Policy)
    def reset(self, obs: CompleteObservation) -> None:
        """Reset the individual components of the submission agent as well as the statistics."""
        SimulationSearchSubmissionAgent.last_top_rhos = []
        SimulationSearchSubmissionAgent.last_top_maze_actions = []

    @override(Policy)
    def seed(self, seed: int) -> None:
        """The seeding method of the base agent

        :param seed: The seed to set.
        """
        # Generate an agent seed and set the seed globally for the model initialization
        self.policy_rng = np.random.RandomState(seed)

    @classmethod
    def build(cls, hydra_config: DictConfig, maze_env: MazeEnv, submission_params: 'SubmissionParams',
              this_directory_path: str, experiment_path: str, ):

        spaces_dict_path = os.path.join(this_directory_path, experiment_path, 'spaces_config.pkl')
        state_dict_path = os.path.join(this_directory_path, experiment_path, 'state_dict.pt')
        # Create the config for the simulated environment
        simulated_env_config = {'_target_': 'maze.core.utils.config_utils.make_env',
                                'env': hydra_config['env'],
                                'wrappers': DictConfig(hydra_config['wrappers'])}

        simulated_env_config['env']['core_env']['power_grid'] = maze_env.wrapped_env
        # Build Simulation-Maze-Environment
        exclude_wrappers = ['maze.core.wrappers.monitoring_wrapper.MazeEnvMonitoringWrapper',
                            'maze.core.wrappers.log_stats_wrapper.LogStatsWrapper',
                            'maze_l2rpn.wrappers.redispatching_controller_wrapper.RedispatchingControllerWrapper']
        simulated_env = prepare_simulated_env(exclude_wrappers=exclude_wrappers, main_env=maze_env,
                                              policy_rng=np.random.RandomState(),
                                              simulated_env=simulated_env_config)
        simulated_env.is_simulated_env = True

        # Build Torch Policy
        torch_policy = SerializedTorchPolicyP36(hydra_config['model'], state_dict_path, spaces_dict_path, device='cpu')

        # Initialize Submission Agent
        return cls(structured_policy=torch_policy, simulated_env=simulated_env,
                   submission_params=submission_params)
