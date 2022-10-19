"""Contains the base class for submittable agents."""
from abc import abstractmethod
from typing import Optional, Tuple, List

from maze.core.agent.policy import Policy
from maze.core.annotations import override
from maze.core.env.base_env import BaseEnv
from maze.core.env.maze_env import MazeEnv
from maze.core.env.maze_state import MazeStateType
from maze.core.env.observation_conversion import ObservationType
from maze.core.env.structured_env import ActorID
from omegaconf import DictConfig

from maze_l2rpn.space_interfaces.action_conversion.dict_unitary import ActionType


class SubmissionPolicy(Policy):

    @abstractmethod
    def act_in_next_step(self) -> bool:
        """If a path is created to the best score let the agent act in the next state event if it is a safe state.

        :return: A bool indicating whether the agent should act in the next state.
        """

    @classmethod
    @abstractmethod
    def build(cls, hydra_config: DictConfig, maze_env: MazeEnv, submission_params: 'SubmissionParams',
              this_directory_path: str, experiment_path: str) -> 'SubmissionPolicy':
        """Build the agent from the hydra config, the maze env and the submission parameters.

        :param hydra_config: The initialized hydra config.
        :param maze_env: The initialized maze_env.
        :param submission_params: The submission parameters.
        :param this_directory_path: The current directory path.
        :param experiment_path: The experiment path.

        :return: The Initialized agent to be used for evaluation.
        """

    @override(Policy)
    def compute_top_action_candidates(self, observation: ObservationType, num_candidates: Optional[int],
                                      maze_state: Optional[MazeStateType], env: Optional[BaseEnv],
                                      actor_id: Optional[ActorID] = None) \
            -> Tuple[List[ActionType], List[float]]:
        """
        Get the top :num_candidates actions as well as the probabilities, q-values, .. leading to the decision.

        :param observation: Current observation of the environment
        :param num_candidates: The number of actions that should be returned. If None all candidates are returned.
        :param maze_state: Current state representation of the environment
                           (only provided if `needs_state()` returns True)
        :param env: The environment instance (only provided if `needs_env()` returns True)
        :param actor_id: ID of actor to query policy for
                         (does not have to be provided if policies dict contains only 1 policy)
        :return: a tuple of sequences, where the first sequence corresponds to the possible actions, the other sequence
                 to the associated scores (e.g, probabilities or Q-values).
        """
        raise NotImplementedError('No need to implement this for submission purposes')

    @override(Policy)
    def needs_env(self) -> bool:
        """Similar to `needs_state`, the policy implementation declares if it operates solely on observations
        (needs_env returns False) or if it also requires the env object in order to compute the action.

        Requiring the env should be regarded as anti-pattern, but is supported for special cases like the MCTS policy,
        which requires cloning support from the environment.

        :return Per default policies return False.
        """
        return True

    @override(Policy)
    def needs_state(self) -> bool:
        """The policy implementation declares if it operates solely on observations (needs_state returns False) or
        if it also requires the state object in order to compute the action.

        Note that requiring the state object comes with performance implications, especially in multi-node distributed
        workloads, where both objects would need to be transferred over the network.
        """
        return True
