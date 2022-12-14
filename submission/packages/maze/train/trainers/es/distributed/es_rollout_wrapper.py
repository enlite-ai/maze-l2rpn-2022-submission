"""Implementation of the rollout generation for ES."""
import itertools
import logging
import time
from typing import TypeVar, Union

import numpy as np
import torch

from maze.core.agent.policy import Policy
from maze.core.agent.torch_model import TorchModel
from maze.core.env.structured_env import StructuredEnv
from maze.core.log_stats.log_stats import LogStatsLevel
from maze.core.log_stats.log_stats_env import LogStatsEnv
from maze.core.wrappers.wrapper import Wrapper
from maze.perception.perception_utils import convert_to_numpy
from maze.train.trainers.es.distributed.es_distributed_rollouts import ESRolloutResult
from maze.train.trainers.es.es_shared_noise_table import SharedNoiseTable
from maze.train.trainers.es.es_utils import get_flat_parameters, set_flat_parameters

logger = logging.getLogger(__name__)


class ESAbortException(Exception):
    """This exception is raised if the current rollout is intentionally aborted."""
    pass


class ESRolloutWorkerWrapper(Wrapper[Union[StructuredEnv, LogStatsEnv]]):
    """The rollout generation is bound to a single worker environment by implementing it as a Wrapper class."""

    def __init__(self,
                 env: Union[StructuredEnv, LogStatsEnv],
                 shared_noise: SharedNoiseTable,
                 agent_instance_seed: int):
        """Avoid calling this constructor directly, use :method:`wrap` instead."""
        super().__init__(env)

        self.shared_noise = shared_noise
        self.abort = False
        self.wrapper_rng = np.random.RandomState(agent_instance_seed)

    def set_abort(self):
        """Abort the rollout (intended to be called from a thread)."""
        self.abort = True

    def clear_abort(self):
        """Clear the abort flag."""
        self.abort = False

    T = TypeVar("T")

    def rollout(self, policy: Union[Policy, TorchModel]) -> None:
        """Use the passed policy to step the environment until it is done.

        This method does not return any results, query the episode statistics instead to process the results.

        :param policy: Multi-step policy encapsulating the policy networks
        """
        observation = self.reset()

        start_time = time.time()

        for _ in itertools.count():
            if self.abort:
                raise ESAbortException()

            with torch.no_grad():
                action = policy.compute_action(
                    observation=observation,
                    actor_id=self.actor_id(),
                    maze_state=self.get_maze_state() if policy.needs_state() else None,
                    env=self if policy.needs_env() else None,
                    deterministic=False)

            observation, reward, done, _ = self.step(convert_to_numpy(action, cast=None, in_place=False))

            if done:
                break

        # reset makes the episode stats available
        self.reset()

        logger.debug(f"Rollout took {(time.time() - start_time) :.1f} seconds")

    def generate_evaluation(self, policy: Union[Policy, TorchModel]) -> ESRolloutResult:
        """Generate a single evaluation rollout.

           :param policy: Multi-step policy encapsulating the policy networks

           :return A result set with a single evaluation rollout
        """
        self.rollout(policy)

        r = ESRolloutResult(is_eval=True)
        aggregator = self.get_stats(LogStatsLevel.EPISODE)
        r.episode_stats.append(aggregator.last_stats)

        return r

    def generate_training(self, policy: Union[Policy, TorchModel], noise_stddev: float) -> ESRolloutResult:
        """Generate a single training sample, consisting of two rollouts, obtained by adding and subtracting the
           same random perturbation vector from the policy.

           :param policy: Multi-step policy encapsulating the policy networks.
           :param noise_stddev: The standard deviation of the applied parameter noise.

           :return A result set with a pair of rollouts generated by adding/subtracting the perturbations
                   (antithetic sampling)
        """
        r = ESRolloutResult(is_eval=False)

        # generate a random noise vector
        noise_idx = self.shared_noise.sample_index(self.wrapper_rng)

        v = noise_stddev * self.shared_noise.get(noise_idx, policy.num_params)
        v = torch.from_numpy(v).to(policy._device)

        # backup the original parameters
        params = get_flat_parameters(policy)

        # prepare the aggregator to receive the result statistics
        aggregator = self.get_stats(LogStatsLevel.EPISODE)

        r.noise_indices.append(noise_idx)

        # --- positive perturbation ---
        # the positive and the negative rollout should be generated under the same conditions
        self.seed(noise_idx)
        set_flat_parameters(policy, params + v)
        self.rollout(policy)
        r.episode_stats.append(aggregator.last_stats)

        # --- negative perturbation ---
        # prepare second rollout
        self.seed(noise_idx)
        set_flat_parameters(policy, params - v)
        self.rollout(policy)
        r.episode_stats.append(aggregator.last_stats)

        # restore the policy state
        set_flat_parameters(policy, params)

        return r
