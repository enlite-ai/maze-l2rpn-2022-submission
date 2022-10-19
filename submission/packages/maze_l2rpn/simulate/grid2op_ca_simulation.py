"""Contains a method for simulating an action on a grid2op env with relaxed parameters for contingency analysis."""
from typing import Tuple, Dict

import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation

from maze_l2rpn.simulate.grid2op_simulate import grid2op_simulate


def grid2op_contingency_analysis_simulate_grid2op_1_7_2_post1(state: CompleteObservation,
                                                              playable_action: PlayableAction,
                                                              ignore_hard_overflow: bool,
                                                              use_forecast: bool) -> \
        Tuple[CompleteObservation, float, bool, Dict]:
    """Simulate an action on the state with custom parameters, that allow all possible contingency events.

    :param state: The complete state to simulate on.
    :param playable_action: The playable outage action to simulate.
    :param ignore_hard_overflow: If set, hard overflows are ignored by the simulation, potentially returning max rho
                                 values above 2.0 (useful to quantify how severe the overflow is).
    :param use_forecast: Either use to forecasts provided by grid2op or inject load and generation of the current state.
    :return: The quadruple of obs, rew, done and info.
    """
    # Record original parameters
    org_MAX_LINE_STATUS_CHANGED = state._obs_env.parameters.MAX_LINE_STATUS_CHANGED
    org_time_before_line_status_actionable_init = state._obs_env.times_before_line_status_actionable_init.copy()
    org_time_before_cooldown_line = state.time_before_cooldown_line.copy()
    org_hard_overflow_threshold = state._obs_env._hard_overflow_threshold

    # Modify the parameters of the simulation env
    state._obs_env._parameters.MAX_LINE_STATUS_CHANGED = state.n_line
    state._obs_env.times_before_line_status_actionable_init = np.zeros_like(org_time_before_line_status_actionable_init)
    state.time_before_cooldown_line = np.zeros_like(state.time_before_cooldown_line)
    if ignore_hard_overflow:
        state._obs_env._hard_overflow_threshold = 9999

    # Simulate the action
    obs, rew, done, info = grid2op_simulate(state, playable_action, use_forcast=use_forecast,
                                            post_state_preparation=False, assert_env_dynamics=False)

    # Reset parameters to original value
    state._obs_env._parameters.MAX_LINE_STATUS_CHANGED = org_MAX_LINE_STATUS_CHANGED
    state._obs_env._hard_overflow_threshold = org_hard_overflow_threshold
    state._obs_env.times_before_line_status_actionable_init = org_time_before_line_status_actionable_init
    state.time_before_cooldown_line = org_time_before_cooldown_line

    return obs, rew, done, info


def grid2op_contingency_analysis_simulate(state: CompleteObservation, playable_action: PlayableAction,
                                          ignore_hard_overflow: bool,
                                          use_forecast: bool) -> \
        Tuple[CompleteObservation, float, bool, Dict]:
    """Simulate an action on the state with custom parameters, that allow all possible contingency events.

    :param state: The complete state to simulate on.
    :param playable_action: The playable outage action to simulate.
    :param ignore_hard_overflow: If set, hard overflows are ignored by the simulation, potentially returning max rho
                                 values above 2.0 (useful to quantify how severe the overflow is).
    :param use_forecast: Either use to forecasts provided by grid2op or inject load and generation of the current state.
    :return: The quadruple of obs, rew, done and info.
    """
    return grid2op_contingency_analysis_simulate_grid2op_1_7_2_post1(state, playable_action,
                                                                     ignore_hard_overflow=ignore_hard_overflow,
                                                                     use_forecast=use_forecast)
