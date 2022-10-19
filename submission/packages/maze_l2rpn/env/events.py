"""Contains events for the grid2op environment."""
from abc import ABC

import numpy as np
from maze.core.log_stats.event_decorators import define_step_stats, define_episode_stats, \
    define_stats_grouping, define_epoch_stats


class ActionEvents(ABC):
    """Event related to actions made by an actor"""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def illegal_action_performed(self, redispatch: bool, reconnect: bool):
        """The event of a agent making a illegal action

        :param redispatch: illegal due to redispatching
        :param reconnect: illegal due to a power-line reconnection
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def ambiguous_action_performed(self):
        """The event of a agent making a ambiguous action
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def noop_action_performed(self):
        """The event of a agent having no effect on the grid
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def topology_action_performed(self):
        """The event of a agent performing a topology change
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def redispatch_action_performed(self):
        """The event of a agent performing a redispatching
        """


class GridEvents(ABC):
    """Event related to a power-grid"""

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    @define_stats_grouping("exception")
    def done(self, exception: str):
        """The event is fired on every done episode, logging the causing exception (or "none")
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    @define_stats_grouping("exception")
    def not_done_exception(self, exception: str):
        """The event is fired whenever an Exception is fired which does not cause a done=True
        """

    @define_epoch_stats(np.mean, output_name="mean_episode_total")
    @define_episode_stats(sum)
    @define_step_stats(len)
    def power_line_overload(self, line_id: int, rho: float):
        """The event of a given power-line overloading

        :param line_id: power line that is currently overflowing
        :param rho: The capacity of the powerline. It is defined at the observed current flow divided by the thermal
        limit of the powerline (no unit)
        """


class RewardEvents(ABC):
    """Event related to grid2op rewards

    Grid2Op grants us access to multiple reward signals by accessing the info dict. These rewards will be forwarded as
    events!
    """

    def l2rpn_reward(self, reward: float):
        """The event of the original l2rpn environment reward

        :param reward: the value
        """

    def other_reward(self, name: str, reward: float, is_kpi: bool):
        """The event of a given reward with a given value

        :param name: reward name
        :param reward: the value
        :param is_kpi: if this score should be included in the KPIs
        """


class ProfilingEvents(ABC):
    """Compute time profiling events."""

    @define_epoch_stats(np.mean, input_name="total_sum", output_name="total_mean")
    @define_epoch_stats(np.mean, input_name="safe_sum", output_name="safe_mean")
    @define_epoch_stats(np.mean, input_name="unsafe_sum", output_name="unsafe_mean")
    @define_episode_stats(np.sum, input_name="total", output_name="total_sum")
    @define_episode_stats(np.sum, input_name="safe", output_name="safe_sum")
    @define_episode_stats(np.sum, input_name="unsafe", output_name="unsafe_sum")
    @define_step_stats(np.sum, input_name="total", output_name="total")
    @define_step_stats(np.sum, input_name="safe", output_name="safe")
    @define_step_stats(np.sum, input_name="unsafe", output_name="unsafe")
    def meta_controller_step_time(self, total: float, safe: float, unsafe: float):
        """Compute times of the meta controller step function `RedispatchingControllerWrapper.step()`

        :param total: time in seconds
        :param safe: time in seconds
        :param unsafe: time in seconds
        """
