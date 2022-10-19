"""Contains a method for simulating a given action on the given state and returning an object from one can continue
simulating."""
import copy
import logging
from datetime import timedelta
from typing import Tuple, Dict, List, Optional

import grid2op
import numpy as np
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation

from maze_l2rpn import APPlY_GRID2OP_BUGFIX_TO_SIMULATION, grid2op_utils, APPLY_RECONNECTION_BUGFIX_TO_SIMULATION
from maze_l2rpn.env.grid2op_step_output_validation import grid2op_step_output_validation

logger = logging.getLogger('SIMULATE')
logger.setLevel(logging.WARNING)


def assert_state_can_be_used_for_simulation(state: CompleteObservation) -> None:
    """Do some simple checks to assert that the state given can be simulated on.

    :param state: The state to simulated on.
    """
    assert state.action_helper is not None
    assert state._obs_env is not None
    assert state._forecasted_inj is not None
    assert state._forecasted_inj[0][0] == state.get_time_stamp()
    assert state._forecasted_inj[0][0] + timedelta(minutes=5) == state._forecasted_inj[1][0]


def check_if_state_matches_sim_env(state: CompleteObservation) -> bool:
    """Check if the state matches the obs_env.

    :param state: The state to check.
    :return: True if it matches and false otherwise.
    """
    # Check obs_env simulation values
    if not np.all(state._obs_env.duration_next_maintenance_init == state.duration_next_maintenance):
        return False
    if not np.all(state._obs_env.time_next_maintenance_init == state.time_next_maintenance):
        return False
    if not np.all(state._obs_env.times_before_line_status_actionable_init == state.time_before_cooldown_line):
        return False
    if not np.all(state._obs_env.times_before_topology_actionable_init == state.time_before_cooldown_sub):
        return False

    if not np.all(state._obs_env._storage_current_charge_init == state.storage_charge):
        return False
    if not np.all(state._obs_env._storage_current_charge == state.storage_charge):
        return False
    if not np.all(state._obs_env._storage_power_init == state.storage_power):
        return False
    if not np.all(state._obs_env._storage_power == state.storage_power):
        return False

    if state._obs_env._nb_time_step_init == state.current_step:
        return False

    # NOTE: Additional values such as attack times need to be updated here.
    return True


def init_state_for_simulation(state: CompleteObservation, use_forcast: bool) -> Optional[Dict[str, np.ndarray]]:
    """Initialize the observation env of the current state for simulation.

    :param state: The state to simulate on.
    :param use_forcast: Specify whether to use the grid2op forcast for simulation (or the current step's forcast again).

    :return: The original forcasting if no forcasting is specified otherwise None.
    """
    if not check_if_state_matches_sim_env(
            state) or state._obs_env.time_stamp is not None:
        # Initialize the init values of the obs_env if it does not match the state or if is the first observation
        #   (to be simulated on)

        state._obs_env.target_dispatch_init = state.target_dispatch.copy()
        state._obs_env.actual_dispatch_init = state.actual_dispatch.copy()

        state._obs_env.duration_next_maintenance_init = state.duration_next_maintenance.copy()
        state._obs_env.time_next_maintenance_init = state.time_next_maintenance.copy()
        state._obs_env.times_before_line_status_actionable_init = state.time_before_cooldown_line.copy()
        state._obs_env.times_before_topology_actionable_init = state.time_before_cooldown_sub.copy()

    # Update the cooldown times in the simulated observation, since there seems to be a bug such the decrease is
    # happening twice (once when the obs env is initialized (in the base observation.simulate method) and once in
    # the step method of the base environment.
    state._obs_env._nb_time_step_init = state.current_step
    state._obs_env.times_before_line_status_actionable_init += 1
    state._obs_env.times_before_topology_actionable_init += 1

    # Copy the observation
    state._obs_env.current_obs_init = state.copy()

    state._obs_env._storage_current_charge_init = state.storage_charge.copy()
    state._obs_env._storage_current_charge = state.storage_charge.copy()
    state._obs_env._storage_power_init = state.storage_power.copy()
    state._obs_env._storage_power = state.storage_power.copy()

    org_forcasting_inj = None
    if not use_forcast:
        # If no forcasting should be used,
        org_forcasting_inj = copy.deepcopy(state._forecasted_inj[1][1])
        state._forecasted_inj = ((state._forecasted_inj[0][0], state._forecasted_inj[0][1]),
                                 (state._forecasted_inj[1][0], state._forecasted_inj[0][1]))

    if 1 in state._forecasted_grid_act and not use_forcast:
        # If no forcast should be used the grid_action must be deleted again since the injections will be changed.
        del state._forecasted_grid_act[1]

    return org_forcasting_inj


def _add_stepping_bug_action(state: CompleteObservation, playable_action: PlayableAction) -> Tuple[bool, List[int]]:
    """Add the stepping bug to the simulation action.
    The stepping bug in the env is concerned with switching off power-lines for no reason due to an indexing error in
    the backend action class. In order to accuratly simulate the behaviour of the env this has to be added to the
    simulation as well since it does not happen there.

    :param state: The current state of the environment.
    :param playable_action: The current playable action of the environment.
    :return: A tuple: First a boolean indicating if the action has been added, second a list of the lines that have
             been modified by the action.
    """
    backend_offline_lines = \
        np.where((state._obs_env._backend_action.current_topo.values[state.line_or_pos_topo_vect] >= 0) & (
                state._obs_env._backend_action.current_topo.values[state.line_ex_pos_topo_vect] >= 0) == False)[0]
    execution_performs_bus_action = np.count_nonzero(grid2op_utils.get_set_bus_action(playable_action)) > 0
    buggy_lines = list(filter(lambda ll: bool(state.line_status[ll]), backend_offline_lines))
    modified_lines = []
    if len(buggy_lines) > 0 and execution_performs_bus_action and APPlY_GRID2OP_BUGFIX_TO_SIMULATION:
        # BColors.print_colored(f'Expecting stepping bug to happen now!!!', BColors.OKBLUE)
        for line in buggy_lines:
            if playable_action.set_line_status[line] == 1:
                continue
            if playable_action.set_bus[state.line_or_pos_topo_vect[line]] != 0 or \
                    playable_action.set_bus[state.line_ex_to_subid[line]] != 0:
                continue
            modified_lines.append(line)
            playable_action.line_set_status = [(line, -1)]

            state.time_before_cooldown_line[line] = 0
            # Do not change the bus when switching off the power lines.
            playable_action.set_bus = [(state.line_or_pos_topo_vect[line], 0), (state.line_ex_pos_topo_vect[line], 0)]
        return True, modified_lines
    return False, modified_lines


def _add_reconnect_bug_action(state: CompleteObservation, playable_action: PlayableAction) -> List[Tuple[int, int]]:
    """Add the reconnection bug fix action.
    This is another bug in the simulation where, when a line is reconnected it is always reconnected to bus 1. However,
    in the actual env it is reconnected to the last set bus. Thus, this has to be fixed by specifying the bus
    explicitly.

    :param state: The current state of the env.
    :param playable_action: The current playable action.
    :return: A list of tuples, of the affected substations and their original cooldown time.
    """
    lines_to_reconnect = np.where(grid2op_utils.get_set_line_status_action(playable_action) == 1)[0]
    affected_substations = list()
    if APPLY_RECONNECTION_BUGFIX_TO_SIMULATION and len(lines_to_reconnect) == 1:
        topo_pos_or = state.line_or_pos_topo_vect[lines_to_reconnect[0]]
        topo_pos_ex = state.line_ex_pos_topo_vect[lines_to_reconnect[0]]
        added_action = False
        if state._obs_env._backend_action.last_topo_registered[topo_pos_or] == 2:
            playable_action.set_bus = [(topo_pos_or, 2)]
            affected_substations.append(state.line_or_to_subid[lines_to_reconnect[0]])
            added_action = True
        if state._obs_env._backend_action.last_topo_registered[topo_pos_ex] == 2:
            playable_action.set_bus = [(topo_pos_ex, 2)]
            affected_substations.append(state.line_ex_to_subid[lines_to_reconnect[0]])
            added_action = True
        if added_action:
            print(f'adding reconnect action bug fix!!!!!!: {playable_action}')
            state._obs_env._parameters.MAX_SUB_CHANGED = 1 + len(affected_substations)
            affected_substations = list(filter(lambda sub_id: state.time_before_cooldown_sub[sub_id] > 0,
                                               affected_substations))
            if len(affected_substations) == 0:
                return affected_substations
            affected_substations_array = np.array(affected_substations)
            affected_substations = list(map(lambda sub_id: (sub_id, state.time_before_cooldown_sub[sub_id] - 1),
                                            affected_substations))
            state.time_before_cooldown_sub[affected_substations_array] = 0

    return affected_substations


def grid2op_simulate(state: CompleteObservation, playable_action: PlayableAction, use_forcast: bool,
                     post_state_preparation: bool, assert_env_dynamics: bool) -> \
        Tuple[CompleteObservation, float, bool, Dict]:
    """Simulate an action on the state with custom parameters, that allow all possible contingency events.

    :param state: The complete state to simulate on.
    :param playable_action: The playable action to simulate.
    :param use_forcast: Specify whether to use the grid2op forcast for simulation (or the current step's forcast again).
    :param post_state_preparation: Specify whether the resulting state should be prepared for further simulation.
                                   NOTE: This involves a deepcopy can therefore be expensive, while at the same time
                                   being redundant in many situations.
    :param assert_env_dynamics: Specify whether the env dynamics should be assertion with a custom method.

    :return: The quadruple of obs, rew, done and info.
    """
    # Make a copy of the _obs_env._backend_action since this should be reset after the simulation.
    org_backend_action_last_topo_registered_values = state._obs_env._backend_action.last_topo_registered.values.copy()
    org_backend_action_last_topo_registered_changed = state._obs_env._backend_action.last_topo_registered.changed.copy()
    org_backend_action_last_topo_registered_index = state._obs_env._backend_action.last_topo_registered.last_index

    # Copy the playable action so no changes are made to the actual object instance
    playable_action = copy.deepcopy(playable_action)
    # OPTIMIZE: ONLY COPY THE PLAYABLE ACTION IF IT IS MODIFIED

    # Assert that the state given can be simulated on.
    assert_state_can_be_used_for_simulation(state)

    affected_substations_by_reconnect_bug = _add_reconnect_bug_action(state, playable_action)

    # Initialize the observation env of the current state.
    org_forcasting_inj = init_state_for_simulation(state=state, use_forcast=use_forcast)

    # Take simulation step
    sim_obs, sim_rew, sim_done, sim_info = _simulate(state, playable_action)

    if len(affected_substations_by_reconnect_bug) > 0 and not sim_done:
        for sub_id, org_cooldown_time in affected_substations_by_reconnect_bug:
            sim_obs.time_before_cooldown_sub[sub_id] = org_cooldown_time

    if not use_forcast:
        state._forecasted_inj = ((state._forecasted_inj[0][0], state._forecasted_inj[0][1]),
                                 (state._forecasted_inj[1][0], org_forcasting_inj))
        # If no forcast is used the forcasting grid action must be deleted again since the injections where changed.
        del state._forecasted_grid_act[1]

    # Do post state preparation for further simulation if not done.
    if not sim_done and post_state_preparation:
        sim_obs._forecasted_inj = copy.deepcopy(state._forecasted_inj)
        sim_obs._forecasted_inj = (
            (sim_obs._forecasted_inj[0][0] + timedelta(minutes=5), sim_obs._forecasted_inj[0][1]),
            (sim_obs._forecasted_inj[1][0] + timedelta(minutes=5), sim_obs._forecasted_inj[1][1]))

        # Copy action helper to the new simulated state
        sim_obs.action_helper = state.action_helper

        # Copy opponent attack information to the new simulated state.
        sim_obs._obs_env._backend_action.last_topo_registered.values[sim_obs.topo_vect == -1] = \
            org_backend_action_last_topo_registered_values[sim_obs.topo_vect == -1]
        sim_obs._obs_env._backend_action.last_topo_registered.changed[sim_obs.topo_vect == -1] = \
            org_backend_action_last_topo_registered_changed[sim_obs.topo_vect == -1]
        opp_space_state = state._obs_env.opp_space_state
        budget_per_timestep = state._obs_env._opponent_budget_per_ts
        sim_obs._obs_env.opp_space_state = (opp_space_state[0] - budget_per_timestep, opp_space_state[1],
                                            opp_space_state[2] - 1, opp_space_state[3] - 1, opp_space_state[4])
        sim_obs._obs_env._opponent_budget_per_ts = copy.deepcopy(budget_per_timestep)

    if not post_state_preparation:
        state._obs_env._backend_action.last_topo_registered.values = org_backend_action_last_topo_registered_values
        state._obs_env._backend_action.last_topo_registered.changed = org_backend_action_last_topo_registered_changed
        state._obs_env._backend_action.last_topo_registered.last_index = org_backend_action_last_topo_registered_index

    if sim_done:
        logger.info(f'simulation failed because reason: {sim_info["exception"]} -- {playable_action}')

    # some of the grid2op-internal rewards return np.float32 instead of float
    sim_rew = float(sim_rew)

    return sim_obs, sim_rew, sim_done, sim_info


def _simulate(state: CompleteObservation, playable_action: PlayableAction) -> \
        Tuple[CompleteObservation, float, bool, Dict]:
    """Simulate an action on the state for the individual grid2op version.

    :param state: The complete state to simulate on.
    :param playable_action: The playable outage action to simulate.
    :return: The quadruple of obs, rew, done and info.
    """
    state._obs_env._parameters.MAX_LINE_STATUS_CHANGED = state.n_line

    sim_obs, sim_rew, sim_done, sim_info = state.simulate(playable_action, time_step=1)

    # Next maintenance time is not decreased in simulation --> Has be done by hand.
    lines_with_scheduled_maintenance = np.where(state.time_next_maintenance > 1)[0]
    sim_obs.time_next_maintenance[lines_with_scheduled_maintenance] = \
        state.time_next_maintenance[lines_with_scheduled_maintenance] - 1

    sim_obs, sim_rew, sim_done, sim_info = grid2op_step_output_validation(sim_obs, sim_rew, sim_done, sim_info,
                                                                          playable_action, None, was_simulated=True)

    return sim_obs, sim_rew, sim_done, sim_info
