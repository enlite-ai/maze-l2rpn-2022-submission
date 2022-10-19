"""Version specific methods for grid2op."""
from typing import Dict, ChainMap, Union, Type, Any

import grid2op
import numpy as np
from grid2op import Environment, Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import BaseReward

from maze.utils.bcolors import BColors

try:
    import lightsim2grid

    lightsim2grid_installed = True
except ImportError:
    lightsim2grid_installed = True
    BColors.print_colored('lightsim2grid backend could not be found. Using pandapower instead',
                          BColors.WARNING)


def init_env(power_grid: str, reward_class: Type, rewards: ChainMap[str, Type[BaseReward]],
             difficulty: str, backend: str, test: bool) -> Environment:
    """Init the grid2op environment.

    :param power_grid: The powergrid to initialize.
    :param reward_class: The reward class.
    :param rewards: The rewards to use.
    :param difficulty: The difficulty level to apply.
    :param backend: The backend to use.
    :param test: Whether this is a test environment or not.

    :return: The initialized environment.
    """
    kwargs = {}
    if reward_class:
        kwargs["reward_class"] = reward_class

    if backend == "lightsim2grid":
        assert lightsim2grid_installed

        if grid2op.__version__ == '1.7.2':
            kwargs['backend'] = lightsim2grid.LightSimBackend()
        else:
            raise NotImplementedError

    if grid2op.__version__ == '1.7.2':
        wrapped_env: grid2op.Environment.Environment = grid2op.make(
            power_grid, other_rewards=dict(rewards), difficulty=str(difficulty),
            test=test, **kwargs)

        # access other_rewards by using the info dict provided by the env.step (info["rewards"][reward_name])
        # update voltage controller
        kwargs_voltage_controller = {'gridobj': type(wrapped_env.backend),
                                     'actionSpace_cls': type(wrapped_env.action_space),
                                     'controler_backend': wrapped_env.backend}
    else:
        raise NotImplementedError

    return wrapped_env


def deepcopy_wrapped_env(src_env: Environment) -> Environment:
    """Prepare deep copy of grid2op environment.

    :param src_env: The environment to be copied.
    :return: A deepcopy of the provided environment.
    """
    if src_env is None:
        return None

    if grid2op.__version__ == '1.7.2':
        return src_env.copy()
    else:
        raise ValueError


def get_sub_topology_of_state(state: CompleteObservation, sub_id: int) -> Dict:
    """Returns the topology of the given substation.

    :param state: The current state.
    :param sub_id: The id of the substation.

    :return: The topology of the substation.
    """

    if grid2op.__version__ in ['1.7.2']:
        return state.sub_topology(sub_id)
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def get_current_step(state: CompleteObservation) -> Union[int, str]:
    """Return the current step of the state.

    :param state: The current maze state.
    :return: The current step if possible
    """
    if grid2op.__version__ in ['1.7.2']:
        return state.current_step
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def get_nb_timesteps_overflow_allowed(state: CompleteObservation) -> int:
    """Retrieve the number of timesteps an overflow is allowed from the state.

    :param state: The current state.
    :return: The number of timesteps an overflow is allowed.
    """
    if grid2op.__version__ in ['1.7.2']:
        return state._obs_env._nb_timestep_overflow_allowed
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def get_parameters(wrapped_env: Environment) -> Parameters:
    """Get the parameters fo the wrapped env by reference!
    :param wrapped_env: The grid2op environment.
    :return: The reference to the parameters.
    """
    if grid2op.__version__ in ['1.7.2']:
        return wrapped_env._parameters
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def get_max_env_steps(wrapped_env: Environment) -> int:
    """Retrieve the maximum steps of the environment.

    :param wrapped_env: The current env.
    :return: The maximal env steps.
    """
    if grid2op.__version__ in ['1.7.2']:
        return wrapped_env.chronics_handler.max_episode_duration()
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def copy_maze_state(state: CompleteObservation) -> CompleteObservation:
    """Copy the maze state with the observation env.

    :param state: The state to copy.
    :return: A copy of the given state.
    """
    if grid2op.__version__ in ['1.7.2']:
        state_copy = state.copy()
        state_copy._obs_env = state._obs_env.copy()
        return state_copy
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def get_set_line_status_action(action: PlayableAction) -> np.ndarray:
    """Get the set line status action of the given playable action.

    :param action: The grid2op action.
    :return: The set line status vector.
    """
    if grid2op.__version__ in ['1.7.2']:
        return action.line_set_status
    else:
        raise NotImplementedError


def get_set_bus_action(action: PlayableAction) -> np.ndarray:
    """Get the set bus action array of given palyable action.

    :param action: The grid2op action.
    :return: The set bus vector.
    """
    if grid2op.__version__ in ['1.7.2']:
        return action.set_bus
    else:
        raise NotImplementedError


def get_disconnected_by_backend(info: Dict[str, Any]) -> np.ndarray:
    """Get a boolean array of lines that have been disconnected by the backend.

    :param info: The info dict returned from the step function.
    :return: A boolean array of all lines disconnected by the backend.
    """
    if grid2op.__version__ in ['1.7.2']:
        return info['disc_lines'] > -1
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def get_lines_disconnected_by_backend(info: Dict[str, Any]) -> np.ndarray:
    """Get the lines that have been disconnected by the backend.

    :param info: The info dict returned from the step function.
    :return: A boolean array of all lines disconnected by the backend.
    """
    return np.where(get_disconnected_by_backend(info))[0]


def get_gen_p(state) -> np.ndarray:
    """Retrieve the generative power flow of the current state.

    :param state: The current state.

    :return: The generative power array.
    """
    if grid2op.__version__ in ['1.7.2']:
        return state.gen_p
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def line_or_bus(state: CompleteObservation) -> np.ndarray:
    """Get the line origin bus configuration of the current state.

    :param state: The current state.
    :return: The line origin bus configuration.
    """
    if grid2op.__version__ in ['1.7.2']:
        return state.line_or_bus
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def line_ex_bus(state: CompleteObservation) -> np.ndarray:
    """Get the line origin bus configuration of the current state.

    :param state: The current state.
    :return: The line origin bus configuration.
    """
    if grid2op.__version__ in ['1.7.2']:
        return state.line_ex_bus
    else:
        raise NotImplementedError(f'Not implemented for grid2op version: {grid2op.__version__}')


def reset_wrapped_env(wrapped_env: Environment) -> None:
    """Reset the wrapped env and the voltage controller if applicable.

    :param wrapped_env: The wrapped grid2op environment.
    """
    wrapped_env.reset()


def sync_current_state_obs_env(grid2op_env: Environment, state: CompleteObservation) -> None:
    """Sync the current state obs env with the wrapped env.
    Specifically the backend action of the obs env needs to be synced, since this is not done by grid2op.

    :param grid2op_env: The grid2op environment to to sync with.
    :param state: The state to update.
    """
    if grid2op_env._backend_action is not None:
        state._obs_env._backend_action.current_topo.values = grid2op_env._backend_action.current_topo.values.copy()
        state._obs_env._backend_action.current_topo.changed = grid2op_env._backend_action.current_topo.changed.copy()
        state._obs_env._backend_action.current_topo.last_index = grid2op_env._backend_action.current_topo.last_index
    if hasattr(grid2op_env, '_opponent_budget_per_ts'):
        state._obs_env._opponent_budget_per_ts = grid2op_env._opponent_budget_per_ts
