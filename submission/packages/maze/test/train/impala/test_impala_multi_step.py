"""Test impala multi step."""
import torch.nn as nn

from maze.core.agent.torch_actor_critic import TorchActorCritic
from maze.core.agent.torch_policy import TorchPolicy
from maze.core.agent.torch_state_critic import TorchSharedStateCritic
from maze.core.wrappers.maze_gym_env_wrapper import GymMazeEnv
from maze.distributions.distribution_mapper import DistributionMapper
from maze.perception.models.built_in.flatten_concat import FlattenConcatPolicyNet, FlattenConcatStateValueNet
from maze.train.parallelization.distributed_actors.distributed_actors import DistributedActors
from maze.train.parallelization.distributed_actors.sequential_distributed_actors import SequentialDistributedActors
from maze.train.parallelization.distributed_actors.subproc_distributed_actors import SubprocDistributedActors
from maze.train.parallelization.vector_env.sequential_vector_env import SequentialVectorEnv
from maze.train.trainers.common.evaluators.rollout_evaluator import RolloutEvaluator
from maze.train.trainers.impala.impala_algorithm_config import ImpalaAlgorithmConfig
from maze.train.trainers.impala.impala_trainer import IMPALA
from maze.utils.timeout import Timeout


def _env_factory():
    return GymMazeEnv("CartPole-v0")


def _policy(env: GymMazeEnv):
    distribution_mapper = DistributionMapper(action_space=env.action_space, distribution_mapper_config={})
    policies = {0: FlattenConcatPolicyNet({'observation': (4,)}, {'action': (2,)}, hidden_units=[16], non_lin=nn.Tanh)}
    critics = {0: FlattenConcatStateValueNet({'observation': (4,)}, hidden_units=[16], non_lin=nn.Tanh)}

    policy = TorchPolicy(networks=policies, distribution_mapper=distribution_mapper, device="cpu")

    critic = TorchSharedStateCritic(networks=critics, obs_spaces_dict=env.observation_spaces_dict,
                                    device="cpu", stack_observations=False)

    return TorchActorCritic(policy=policy, critic=critic, device="cpu")


def _algorithm_config():
    eval_env = SequentialVectorEnv([_env_factory for _ in range(2)], logging_prefix='eval')

    return ImpalaAlgorithmConfig(
        n_epochs=2,
        epoch_length=2,
        queue_out_of_sync_factor=2,
        patience=15,
        n_rollout_steps=20,
        lr=0.0005,
        gamma=0.98,
        policy_loss_coef=1.0,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.0,
        device="cpu",
        vtrace_clip_pg_rho_threshold=1,
        vtrace_clip_rho_threshold=1,
        num_actors=1,
        actors_batch_size=5,
        critic_burn_in_epochs=0,
        rollout_evaluator=RolloutEvaluator(eval_env=eval_env, n_episodes=1,
                                           model_selection=None, deterministic=True)
    )


def _train_function(train_actors: DistributedActors, algorithm_config: ImpalaAlgorithmConfig) -> IMPALA:
    impala = IMPALA(model=_policy(train_actors.env_factory()),
                    rollout_generator=train_actors,
                    evaluator=algorithm_config.rollout_evaluator,
                    algorithm_config=algorithm_config,
                    model_selection=None)

    impala.train(n_epochs=algorithm_config.n_epochs)

    return impala


def test_impala_multi_step_dummy():
    algorithm_config = _algorithm_config()
    train_actors = SequentialDistributedActors(_env_factory, _policy(_env_factory()).policy,
                                               n_rollout_steps=algorithm_config.n_rollout_steps,
                                               n_actors=algorithm_config.num_actors,
                                               batch_size=algorithm_config.actors_batch_size,
                                               actor_env_seeds=[1234 for _ in range(algorithm_config.num_actors)])
    impala = _train_function(train_actors, algorithm_config)
    assert isinstance(impala, IMPALA)


def test_impala_multi_step_distributed():
    algorithm_config = _algorithm_config()
    train_actors = SubprocDistributedActors(_env_factory, _policy(_env_factory()).policy,
                                            n_rollout_steps=algorithm_config.n_rollout_steps,
                                            n_actors=algorithm_config.num_actors,
                                            batch_size=algorithm_config.actors_batch_size,
                                            queue_out_of_sync_factor=algorithm_config.queue_out_of_sync_factor,
                                            start_method="forkserver",
                                            actor_agent_seeds=[4321 for _ in range(algorithm_config.num_actors)],
                                            actor_env_seeds=[1234 for _ in range(algorithm_config.num_actors)])
    with Timeout(seconds=30):
        impala = _train_function(train_actors, algorithm_config)
    assert isinstance(impala, IMPALA)
