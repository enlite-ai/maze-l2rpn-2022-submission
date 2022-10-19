"""Contains the core env for the l2rpn-challenge."""
import os
import pickle
from collections import ChainMap
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Union, Mapping, Type, Optional

import grid2op
import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import BaseReward
from maze.core.annotations import override
from maze.core.env.core_env import CoreEnv
from maze.core.env.reward import RewardAggregatorInterface
from maze.core.env.simulated_env_mixin import SimulatedEnvMixin
from maze.core.env.structured_env import ActorID
from maze.core.events.pubsub import Pubsub
from maze.core.rendering.renderer import Renderer
from maze.core.utils.factory import Factory
from maze.utils.bcolors import BColors
from omegaconf import DictConfig, OmegaConf

from maze_l2rpn.env.events import ActionEvents, GridEvents, RewardEvents, ProfilingEvents
from maze_l2rpn.env.kpi_calculator import Grid2OpKpiCalculator
from maze_l2rpn.env.l2rpn_renderer import L2RPNRenderer


@dataclass
class L2RPNSeedInfo:
    """Represents seed information of the Grid2Op env."""

    env_index: int
    """Index of the environment within the multi-environment."""

    chronic_id: int
    """Index of the Grid2Op chronics data."""

    random_seed: int
    """Seed for the pseudo-random generators."""

    fast_forward: int
    """Applied to Grid2Op fast_forward_chronics() method."""

    actions: Tuple[Any, ...]
    """Sequence of actions applied on env reset()."""

    def __hash__(self):
        return hash(tuple([self.env_index, self.chronic_id, self.random_seed, self.fast_forward, tuple(self.actions)]))


class Grid2OpCoreEnvironment(CoreEnv):
    """Core env for the l2rpn challenge.

    :param power_grid: The name fo the grid2op environment setting.
    :param difficulty: The difficulty level of the env.
    :param reward: The reward configuration.
    :param reward_aggregator: The reward aggregator used.
    :param chronics_config: The configuration of the chronics.
    """

    def __init__(self,
                 power_grid: Union[str, grid2op.Environment.Environment],
                 difficulty: Union[int, str],
                 reward: Union[Type[BaseReward], str, Mapping[str, Any]],
                 reward_aggregator: Union[RewardAggregatorInterface, str, Mapping[str, Any]],
                 chronics_config: Union[DictConfig, Dict]):
        super().__init__()
        self._init_chronics(chronics_config)

        # the external grid2op environment
        self.wrapped_env: Optional[grid2op.Environment.Environment] = None

        # init pubsub for event to reward routing
        self.pubsub = Pubsub(self.context.event_service)

        # init reward function(s)
        self._init_reward(reward)

        # setup environment
        self._init_env(power_grid, difficulty)
        self._setup_env()

        # init reward aggregator
        self._init_reward_aggregator(reward_aggregator)

        # KPIs calculation
        self.kpi_calculator = Grid2OpKpiCalculator()

        # support seeding by replay to a certain seed
        self._seed_info: Optional[L2RPNSeedInfo] = None

        # Rendering
        self.renderer = L2RPNRenderer(self.wrapped_env.observation_space)

        self._current_state: Optional[CompleteObservation] = None

    def _setup_env(self) -> None:
        """Setup environment.
        """
        self.action_events = self.pubsub.create_event_topic(ActionEvents)
        self.grid_events = self.pubsub.create_event_topic(GridEvents)
        self.reward_events = self.pubsub.create_event_topic(RewardEvents)
        self.profiling_events = self.pubsub.create_event_topic(ProfilingEvents)

    def _init_env(self, power_grid: Union[str, grid2op.Environment.Environment], difficulty: Union[int, str]) -> None:
        """Instantiate power grid environment from problem instance identifier.

        :param power_grid: Power grid problem instance identifier or instance.
        :param difficulty: The difficulty level of the env.
        """
        possible_difficulties = [0, 1, 2, '0', '1', '2', 'competition_no_subcooldown', 'competition']
        assert difficulty in possible_difficulties, f'The difficulty should be in {possible_difficulties}'
        if isinstance(power_grid, str):
            # combine all reward classes in a single dict
            rewards = ChainMap(self.reward_classes, self.kpi_classes)

            try:
                import lightsim2grid
                backend = lightsim2grid.LightSimBackend()
                backend = {'backend': backend}
            except ImportError:
                BColors.print_colored('Lightsim2grid backend could not be found. Using Pandas instead', BColors.WARNING)
                backend = {}
            self.wrapped_env: grid2op.Environment.Environment = grid2op.make(power_grid,
                                                                             reward_class=self.reward_class,
                                                                             other_rewards=dict(rewards),
                                                                             difficulty=str(difficulty), **backend)

            # access other_rewards by using the info dict provided by the env.step (info["rewards"][reward_name])

        else:
            self.wrapped_env: grid2op.Environment.Environment = power_grid
            BColors.print_colored('env difficulty could not be applied since the env was passed as an instance',
                                  color=BColors.WARNING)

    def _init_reward(self, reward: Union[Type[BaseReward], str, Mapping[str, Any]]) -> None:
        """Instantiate rewards and "KPIs" for the environment.

        :param reward: The reward to use.
        """
        self.reward_classes = dict()
        self.kpi_classes = dict()

        if isinstance(reward, type):
            self.reward_class = reward

        elif isinstance(reward, str):
            self.reward_class = Factory(base_type=BaseReward).type_from_name(reward)

        # handle mapping type
        else:
            self.reward_class = Factory(base_type=BaseReward).type_from_name(reward["_target_"])

            if "rewards" in reward:
                for i, v in enumerate(reward["rewards"]):
                    _reward_class = Factory(base_type=BaseReward).type_from_name(v["_target_"])

                    # check if reward is a kpi score in reality
                    if v.get("kpi"):
                        self.kpi_classes[v["name"]] = _reward_class

                    # reward is not a kpi
                    if "name" in v:
                        self.reward_classes[v["name"]] = _reward_class
                    else:
                        self.reward_classes[f"reward_{i + 1}"] = _reward_class

    def _init_reward_aggregator(self,
                                reward_aggregator: Union[RewardAggregatorInterface, str, Mapping[str, Any]]) -> None:
        """Instantiate reward aggregator.

        :param reward_aggregator: The reward aggregator object.
        """
        self.reward_aggregator = Factory(base_type=RewardAggregatorInterface).instantiate(reward_aggregator)
        self.pubsub.register_subscriber(self.reward_aggregator)

    def _init_chronics(self, chronics_config: Union[DictConfig, Dict]) -> None:
        """Instantiate the chronics config if a file is given."""
        if not isinstance(chronics_config, DictConfig):
            assert isinstance(chronics_config, dict)
            chronics_config = DictConfig(chronics_config)

        # load chronics config file containing a list [(chronics_id, starting_step), ...]
        if chronics_config.from_file is not None:
            assert os.path.exists(chronics_config.from_file), \
                f"Provided chronics config file '{chronics_config.from_file}' not found!"
            with open(chronics_config.from_file, 'rb') as file_handler:
                starting_points = pickle.load(file_handler)
                chronics_config = OmegaConf.to_container(chronics_config, resolve=True)
                chronics_config["starting_points"] = starting_points
                chronics_config = DictConfig(chronics_config)

        self.chronics_config = chronics_config

    @override(CoreEnv)
    def step(self, execution: PlayableAction) -> Tuple[CompleteObservation, np.array, bool, Dict[Any, Any]]:
        """Just passes the relevant information to the grid2op wrapped env.

        :param execution: Environment action to take.
        :return: state, reward, done, info
        """
        raise NotImplementedError("The core env should never be stepped!")

    @override(CoreEnv)
    def get_maze_state(self) -> CompleteObservation:
        """Return current state of the environment.
        """
        return self._current_state

    def set_maze_state(self, maze_state: CompleteObservation) -> None:
        """Set the current state of the environment.

        This is necessary for the wrapper stack to work in case we do not actually have the env to step through like in
        like in the evaluation (script) case.
        """
        self._current_state = maze_state

    @override(CoreEnv)
    def get_kpi_calculator(self) -> Grid2OpKpiCalculator:
        """KPIs are supported."""
        return self.kpi_calculator

    @override(CoreEnv)
    def reset(self) -> CompleteObservation:
        """Resets the environment"""

        # set seed
        if self._seed_info:
            self.wrapped_env.set_id(self._seed_info.chronic_id)
            self.wrapped_env.seed(self._seed_info.random_seed)
            # necessary for MultiMixEnv only: seed() method does not handle the env sub-index
            self.wrapped_env.env_index = self._seed_info.env_index

        # reset grid2op env
        # NOTE: WRAPPED ENV SHOULD NOT BE RESET AS THIS CAN LEAD TO PROBLEMS
        # self.wrapped_env.reset()

        # reset historic values in renderer
        self.renderer.reset_values()

        # reset episode according to provided replay
        if self._seed_info:

            # proceed with noops
            for _ in range(self._seed_info.fast_forward):
                self.wrapped_env.step(self.wrapped_env.action_space({}))

            # proceed with action sequence
            if self._seed_info.actions:
                for action in self._seed_info.actions:
                    obs, reward, done, info = self.wrapped_env.step(action)
                    assert not done, f"trajectory seed could not be applied, " \
                                     f"env was done before the end of the initial action sequence\n" \
                                     f"Info: {info}"

        self._current_state = self.wrapped_env.get_obs()

        return self._current_state

    @override(SimulatedEnvMixin)
    def clone_from(self, env: "Grid2OpCoreEnvironment") -> None:
        """Reset this env to the given state."""
        # No clone from necessary
        self._current_state = env._current_state

    @override(CoreEnv)
    def close(self) -> None:
        """No additional cleanup necessary."""
        pass

    @override(CoreEnv)
    def seed(self, seed: L2RPNSeedInfo) -> None:
        """Seeds the environment"""
        self._seed_info = seed

    def get_current_seed(self) -> L2RPNSeedInfo:
        """Returns the currently used seed."""
        return self._seed_info

    @override(CoreEnv)
    def get_serializable_components(self) -> Dict[str, Any]:
        """Serializable all components used within the grid2op env.

        :return: a list of serialized components
        """
        return {}

    @override(CoreEnv)
    def get_renderer(self) -> Renderer:
        """Wrapped renderer for the grid2op env."""
        return self.renderer

    @override(CoreEnv)
    def actor_id(self) -> ActorID:
        """Currently implemented as single policy, single actor env.

        :return: Actor and id, in this case always 0,0
        """
        return ActorID(step_key=0, agent_id=0)

    @property
    @override(CoreEnv)
    def agent_counts_dict(self) -> Dict[Union[str, int], int]:
        """Returns the count of agents for individual sub-steps (or -1 for dynamic agent count).

        As this is a single-step single-agent environment, in which 1 agent gets to act during sub-step 0,
        we return {0: 1}.
        """
        return {0: 1}

    @override(CoreEnv)
    def is_actor_done(self) -> bool:
        """The actors of this env are never done.

        :return: In this case always False
        """
        return False
