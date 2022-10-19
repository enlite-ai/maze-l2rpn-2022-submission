"""Contains a method for asserting the output of the grid2op step method."""
import logging
from datetime import timedelta
from logging import Logger
from typing import Dict, Any, Optional, Union

import grid2op
import lightsim2grid
import numpy as np
from grid2op import Environment
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation

from maze_l2rpn import grid2op_utils
from maze_l2rpn.env.events import GridEvents
from maze_l2rpn.env.l2rpn_seed_info import L2RPNSeedInfo
from maze_l2rpn.grid2op_utils import get_nb_timesteps_overflow_allowed, get_lines_disconnected_by_backend, \
    get_current_step
from maze.utils.bcolors import BColors

logger = logging.getLogger('STEP_ASSERTION')
logger.setLevel(logging.DEBUG)


def compare_simulation_with_actual_step_output(state: CompleteObservation, reward: Optional[float], done: bool,
                                               info: Optional[Dict[str, Any]], sim_state: CompleteObservation,
                                               sim_reward: float, sim_done: bool, sim_info: Dict[str, Any],
                                               last_state: Optional[CompleteObservation],
                                               execution: Union[PlayableAction, str],
                                               grid_events: Optional[GridEvents],
                                               seed_info: L2RPNSeedInfo, logger: Logger) -> None:
    """If the debugging flag is set, compare the simulation result of the execution on the last step with the result
    of calling the wrapped.step function.

    :param state: The state as a result of calling the step method.
    :param reward: The reward as a result of calling the step method.
    :param done: The done as a result of calling the step method.
    :param info: The info as a result of calling the step method.
    :param sim_state: The state as a result of calling the simulate method.
    :param sim_reward: The reward as a result of calling the simulate method.
    :param sim_done: The done as a result of calling the simulate method.
    :param sim_info: The info as a result of calling the simulate method.
    :param last_state: The previous state.
    :param execution: The execution used to step the env.
    :param grid_events: The optional grid events.
    :param seed_info: The current seed info.
    :param logger: The logger to use.
    """

    if not done and not sim_done and not np.all(sim_state.line_status == state.line_status):

        different_lines = np.where(sim_state.line_status != state.line_status)[0]
        if np.all(sim_state.rho[different_lines] > 1.8) or np.all(state.rho[different_lines] > 1.8):
            # Simulation error when calculating the cascading failure
            return

        if info is not None and np.all(info['disc_lines'][different_lines] >= 3):
            # if lines are part of a cascading failure in the 3rd iterations than this might be due to the
            #   simulation not performing enough iterations
            return

        if last_state is not None and np.all(last_state.timestep_overflow[different_lines] == 3) and \
                np.all(sim_state.rho[different_lines] > 0.95):
            # if lines have been overflown for 3 steps and are close to being overflown in the simulated state,
            # they can have been switched off in the real state.
            return

        attacked_lines = []
        if info is not None and 'opponent_attack_line' in info:
            attacked_lines = list(np.where(info['oppend_attack_lines'])[0])
            if np.all(info['opponent_attack_line'][different_lines]):
                # Lines were attacked... this could not be simulated
                return

        if hasattr(state._obs_env, 'opp_space_state') and state._obs_env.opp_space_state is not None \
                and state._obs_env.opp_space_state[4] is not None:
            attacked_lines += list(np.where(state._obs_env.opp_space_state[4]._lines_impacted)[0])
            attacked_lines = list(set(attacked_lines))
            if np.all(state._obs_env.opp_space_state[4]._lines_impacted[different_lines]):
                # Lines were attacked... this could not be simulated
                return

        logger.warning(BColors.format_colored(
            f'Simulation did not predict the line status correctly! Offline lines in simulation: '
            f'{np.where(~sim_state.line_status)[0]} vs actual: {np.where(~state.line_status)[0]} '
            f'and lines offline before: {np.where(~last_state.line_status)[0] if last_state is not None else None}'
            , BColors.WARNING))

        max_overflown_in_last_step = get_nb_timesteps_overflow_allowed(
            state) == state.timestep_overflow

        lines_disconnected_by_backend = get_lines_disconnected_by_backend(sim_info)
        warning_txt = (
            f'Lines offline before: {np.where(~last_state.line_status)[0] if last_state is not None else None}, '
            f'num_lines: {len(state.line_status)}, at step: {get_current_step(state)}, '
            f'with seed: {seed_info})')
        if last_state is not None:
            lines_overflown = np.where(last_state.rho > 1.0)[0]
            zip_overflown = list(zip(list(lines_overflown), list(last_state.timestep_overflow[lines_overflown])))
            warning_txt += (
                f'\nBEFORE:'
                f'\noverflown (line, nb_overflown): {zip_overflown}'
            )
        warning_txt += (
            f'\nSIMULATION:'
            f'\n\tlines_offline_now: {np.where(~sim_state.line_status)[0]}'
            f'\n\tlines_went_maintenance: {np.where(sim_state.time_next_maintenance == 1)[0]}'
            f'\n\tlines_max_overflown: {np.where(~((0.001 < sim_state.rho[max_overflown_in_last_step]) & (sim_state.rho[max_overflown_in_last_step] < 1.0)))[0]}'
            f'\n\tlines_disconnected_by_backend: {list(zip(lines_disconnected_by_backend, sim_info["disc_lines"][lines_disconnected_by_backend]))}'
            f'\n\tis_illegal: {sim_info["is_illegal"]}'
            f'\n\tis_ambiguous: {sim_info["is_illegal"]}'
            f'\n\texception: {sim_info["exception"]}'
            f'\n\trho for lines that differ: {list(zip(different_lines, sim_state.rho[different_lines]))}'
            f'\nACTUAL:'
            f'\n\tlines_offline_now: {np.where(~state.line_status)[0]}'
            f'\n\tlines_went_maintenance: {np.where(state.time_next_maintenance == 1)[0]}'
            f'\n\tlines_max_overflown: {np.where(~((0.001 < state.rho[max_overflown_in_last_step]) & (state.rho[max_overflown_in_last_step] < 1.0)))[0]}'
            f'\n\tlines under attack: {attacked_lines}')
        if info is not None:
            warning_txt += (
                f'\n\tlines_disconnected_by_backend: {list(zip(np.where(info["disc_lines"] > -1)[0], info["disc_lines"][np.where(info["disc_lines"] > -1)[0]]))}'
                f'\n\tis_illegal: {info["is_illegal"]}'
                f'\n\tis_ambiguous: {info["is_illegal"]}'
                f'\n\texception: {info["exception"]}')
        warning_txt += (
            f'\n\trho for lines that differ: {list(zip(different_lines, state.rho[different_lines]))}'
            f'\nACTION'
            f'\n{execution}')

        logger.warning(BColors.format_colored(warning_txt, BColors.WARNING))

    if done and grid_events is not None:
        grid_events.simulation_did_not_predict_done(
            int(done and not sim_done and len(info['exception']) > 0))
    if done and not sim_done and info is not None and len(info['exception']) > 0 and sim_state.rho.max() < 1.9:
        logger.warning(BColors.format_colored(f'Env done but simulation did not predict a done. '
                                              f'reason: {info["exception"]} - '
                                              f'max rho of sim state: {sim_state.rho.max()} - \n{execution}',
                                              BColors.WARNING))
    if sim_done and not done and state.rho.max() < 1.9:
        logger.warning(BColors.format_colored(f'Not done, but simulation predicted done. '
                                              f'reason: {sim_info["exception"]}, '
                                              f'max rho of state: {state.rho.max()}', BColors.WARNING))
    if not done:
        max_rho_diff = np.abs(sim_state.rho.max() - state.rho.max())
        if grid_events is not None:
            grid_events.simulation_error(max_rho_diff)
