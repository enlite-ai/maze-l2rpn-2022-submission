"""Sequential rollout runner for running envs and agents in the local process."""
from tqdm import tqdm

from maze.core.annotations import override
from maze.core.log_events.log_events_writer_registry import LogEventsWriterRegistry
from maze.core.log_events.log_events_writer_tsv import LogEventsWriterTSV
from maze.core.log_stats.log_stats import register_log_stats_writer
from maze.core.log_stats.log_stats_writer_console import LogStatsWriterConsole
from maze.core.rollout.rollout_runner import RolloutRunner
from maze.core.trajectory_recording.writers.trajectory_writer_file import TrajectoryWriterFile
from maze.core.trajectory_recording.writers.trajectory_writer_registry import TrajectoryWriterRegistry
from maze.core.utils.factory import ConfigType, CollectionOfConfigType
from maze.core.wrappers.log_stats_wrapper import LogStatsWrapper
from maze.core.wrappers.trajectory_recording_wrapper import TrajectoryRecordingWrapper
from maze.utils.bcolors import BColors


class SequentialRolloutRunner(RolloutRunner):
    """Runs rollout in the local process. Useful for short rollouts or debugging.

    Trajectory, event logs and stats are recorded into the working directory managed by hydra (provided
    that the relevant wrappers are present.)

    :param n_episodes: Count of episodes to run
    :param max_episode_steps: Count of steps to run in each episode (if environment returns done, the episode
                                will be finished earlier though)
    :param deterministic: Deterministic or stochastic action sampling.
    :param record_trajectory: Whether to record trajectory data
    :param record_event_logs: Whether to record event logs
    """

    def __init__(self,
                 n_episodes: int,
                 max_episode_steps: int,
                 deterministic: bool,
                 record_trajectory: bool,
                 record_event_logs: bool,
                 render: bool):
        super().__init__(n_episodes=n_episodes, max_episode_steps=max_episode_steps, deterministic=deterministic,
                         record_trajectory=record_trajectory, record_event_logs=record_event_logs)
        if render:
            assert record_trajectory, "Rendering is supported only when trajectory recording is enabled."

        self.render = render
        self.progress_bar = None

    @override(RolloutRunner)
    def run_with(self, env: ConfigType, wrappers: CollectionOfConfigType, agent: ConfigType):
        """Run the rollout sequentially in the main process."""
        env_seeds = self.maze_seeding.get_explicit_env_seeds(self.n_episodes)
        agent_seeds = self.maze_seeding.get_explicit_agent_seeds(self.n_episodes)
        env, agent = self.init_env_and_agent(env_config=env, wrappers_config=wrappers, input_dir=self.input_dir,
                                             max_episode_steps=self.max_episode_steps, agent_config=agent)

        # Set up the wrappers
        # Hydra handles working directory
        register_log_stats_writer(LogStatsWriterConsole())
        if not isinstance(env, LogStatsWrapper):
            env = LogStatsWrapper.wrap(env, logging_prefix="rollout_data")
        if self.record_event_logs:
            LogEventsWriterRegistry.register_writer(LogEventsWriterTSV(log_dir="./event_logs"))
        if self.record_trajectory:
            TrajectoryWriterRegistry.register_writer(TrajectoryWriterFile(log_dir="./trajectory_data"))
            if not isinstance(env, TrajectoryRecordingWrapper):
                env = TrajectoryRecordingWrapper.wrap(env)

        actual_number_of_episodes = min(len(env_seeds), self.n_episodes)
        if actual_number_of_episodes < self.n_episodes:
            BColors.print_colored(f'Only {len(env_seeds)} explicit seed(s) given, thus the number of episodes changed '
                                  f'from: {self.n_episodes} to {actual_number_of_episodes}.', BColors.WARNING)
        self.progress_bar = tqdm(desc="Episodes done", unit=" episodes", total=actual_number_of_episodes)
        RolloutRunner.run_interaction_loop(env, agent, actual_number_of_episodes, render=self.render,
                                           after_reset_callback=lambda: self.update_progress(),
                                           env_seeds=env_seeds, agent_seeds=agent_seeds,
                                           deterministic=self.deterministic)
        self.progress_bar.close()
        env.write_epoch_stats()

    def update_progress(self):
        """Called on episode end to update a simple progress indicator."""
        self.progress_bar.update()
