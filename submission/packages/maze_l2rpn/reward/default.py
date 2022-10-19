"""Contains a default reward aggregator"""
from abc import ABC
from typing import List, Type, Optional

from maze.core.env.reward import RewardAggregatorInterface
from maze.core.env.maze_state import MazeStateType

from maze_l2rpn.env.events import RewardEvents


class RewardAggregator(RewardAggregatorInterface):
    """Default event aggregation object dealing with rewards.

    :param reward_scale: global reward scaling factor
    """
    def __init__(self, reward_scale: float):
        super().__init__()
        self.reward_scale = reward_scale

    def get_interfaces(self) -> List[Type[ABC]]:
        """Provides a list of reward relevant event interfaces.
        
        :return: List of event interfaces.
        """""
        return [RewardEvents]

    def summarize_reward(self, maze_state: Optional[MazeStateType] = None) -> float:
        """Summarizes all reward relevant events to a scalar value.

        :param maze_state: Not used by this reward aggregator.
        :return: The accumulated scalar reward.
        """
        total_reward = 0.0

        # process the l2rpn environment reward
        l2rpn_rewards = [ev for ev in self.query_events([RewardEvents.l2rpn_reward])]
        assert len(l2rpn_rewards) == 1
        total_reward += l2rpn_rewards[0].reward

        return total_reward * self.reward_scale
