"""Contains performance unit tests."""
import glob
from typing import Dict

import numpy as np
import pytest
from maze.test.shared_test_utils.run_maze_utils import run_maze_job
from maze.utils.tensorboard_reader import tensorboard_to_pandas
from maze.utils.timeout import Timeout

# Configurations to be tested
trainings = [
    # PPO
    [180, {"algorithm": "ppo", "algorithm.n_epochs": "3", "algorithm.rollout_evaluator.n_episodes": "0",
           "env": "gym_env", "env.name": "CartPole-v0"}],
    # A2C
    [180, {"algorithm": "a2c", "algorithm.n_epochs": "5", "algorithm.rollout_evaluator.n_episodes": "0",
           "env": "gym_env", "env.name": "CartPole-v0"}],
    # IMPALA
    [180, {"algorithm": "impala", "algorithm.n_epochs": "5", "algorithm.rollout_evaluator.n_episodes": "0",
           "env": "gym_env", "env.name": "CartPole-v0"}],
    # SAC
    [180, {"algorithm": "sac", "algorithm.n_epochs": "30", "algorithm.rollout_evaluator.n_episodes": "0",
           "env": "gym_env", "env.name": "CartPole-v0",
           "model": "vector_obs", "critic": "template_state_action"}],
    # ES
    [180, {"algorithm": "es", "algorithm.n_epochs": "100", "algorithm.n_rollouts_per_update": "20",
           "env": "gym_env", "env.name": "CartPole-v0"}],
]


@pytest.mark.longrun
@pytest.mark.parametrize("target_reward, hydra_overrides", trainings)
def test_train(hydra_overrides: Dict[str, str], target_reward: float):
    # run training
    with Timeout(seconds=300):
        run_maze_job(hydra_overrides, config_module="maze.conf", config_name="conf_train")

    # load tensorboard log
    tf_summary_files = glob.glob("*events.out.tfevents*")
    assert len(tf_summary_files) == 1, f"expected exactly 1 tensorflow summary file {tf_summary_files}"
    events_df = tensorboard_to_pandas(tf_summary_files[0])

    # check if target reward was reached
    max_mean_reward = np.nanmax(np.asarray(events_df.loc["train_BaseEnvEvents/reward/mean"]))
    assert max_mean_reward >= target_reward
