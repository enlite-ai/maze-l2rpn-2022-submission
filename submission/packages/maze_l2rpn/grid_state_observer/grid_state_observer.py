"""Contains the grid state observer base class."""
import copy
import logging
from typing import Dict, Optional, List, Tuple

import grid2op
import numpy as np
from grid2op.Observation import CompleteObservation
from gym import spaces
from maze.core.utils.factory import Factory
from maze.utils.bcolors import BColors

from maze_l2rpn.grid_state_observer.grid_state_observer_types import Contingency, GridElement, ViolationLevel, \
    GridElementType, ExpectedContingency
from maze_l2rpn.grid_state_observer.state_tests.state_test_interface import StateTestInterface
from maze_l2rpn.grid_state_observer.violations.divirging_power_flow_violation import DivergingPowerFlowViolation
from maze_l2rpn.grid_state_observer.violations.violation import Violations
from maze_l2rpn.simulate.grid2op_ca_simulation import grid2op_contingency_analysis_simulate

logger = logging.getLogger('GSO')
logger.setLevel(logging.WARNING)


class GridStateObserver:
    """The Grid-State-Observer (GSO) class analyses the current state w.r.t. to given state tests. Furthermore, expected
    outages such as scheduled maintenance are simulated and the resulting state tested. Finally, a contingency list can
    also be given where each contingency is simulated and the resulting state validated.

    :param core_env: The grid core environment.
    :param state_tests: An optional list of state tests to be applied to the current state.
    :param expected_outage_lookahead: An integer specifying the lookahead of scheduled maintenance. If a state is
                                      encountered where a scheduled maintenance occurs in time steps less than the
                                      specified value an outage of the maintenance is simulated and the resulting state
                                      validated with the post_expected_outage_validation list.
    :param post_expected_outage_state_tests: An optional list of state tests to be applied to the resulting state after
                                             simulating the expected outage event.
    :param contingency_list: An optional list of contingencies, where for each element in the list, the contingency is
                             simulated and the resulting state tested w.r.t. to the post_contingency_state_validation
                             list.
    :param post_contingency_state_tests: An optional list of state tests to be applied to the resulting state after
                                         simulating the outage of the contingencies defined in the contingency list.
    :param calculate_post_contingency_if_state_is_not_safe: A bool specifying whether to preform the full contingency
                                                            analysis if the state validation already found violations.
    """

    gso_observation_space = spaces.Dict(
        {"is_safe_state": spaces.Box(low=np.float32(0), high=np.float32(1), shape=(1,),
                                     dtype=np.float32),
         "n_violations_severe": spaces.Box(low=np.float32(0), high=np.finfo(np.float32).max, shape=(1,),
                                           dtype=np.float32),
         "n_violations_critical": spaces.Box(low=np.float32(0), high=np.finfo(np.float32).max, shape=(1,),
                                             dtype=np.float32)
         }
    )

    def __init__(self, wrapped_env: grid2op.Environment.Environment,
                 state_tests: Optional[List[StateTestInterface]],
                 expected_outage_lookahead: int,
                 post_expected_outage_state_tests: Optional[List[StateTestInterface]],
                 calculate_post_contingency_if_state_is_not_safe: bool):

        # Assertions
        assert state_tests is None or len(state_tests) > 0
        assert post_expected_outage_state_tests is None or len(post_expected_outage_state_tests) > 0

        assert (post_expected_outage_state_tests is None and expected_outage_lookahead == 0) or \
               (post_expected_outage_state_tests is not None and expected_outage_lookahead > 0), \
            f'{post_expected_outage_state_tests} and {expected_outage_lookahead}'

        # Init state validation
        self._state_tests = [] if state_tests is None else state_tests
        self._state_tests = [Factory(base_type=StateTestInterface).instantiate(st) for st in self._state_tests]
        # Init post expected outage validation
        self._post_expected_outage_state_tests = [] if post_expected_outage_state_tests is None else \
            post_expected_outage_state_tests
        self._post_expected_outage_state_tests = \
            [Factory(base_type=StateTestInterface).instantiate(st) for st in self._post_expected_outage_state_tests]

        self._expected_outage_lookahead = expected_outage_lookahead
        self._calculate_post_contingency_if_state_is_not_safe = calculate_post_contingency_if_state_is_not_safe

        self._wrapped_env_action_space = None
        if len(self._post_expected_outage_state_tests) > 0:
            self._wrapped_env_action_space = wrapped_env.action_space

        self._latest_state_hash = None
        self._latest_is_safe_state = True

        self._latest_violations: Violations = Violations()
        self._latest_violation_counts: Dict[ViolationLevel, int] = dict()

    def __call__(self, state: CompleteObservation) -> bool:
        """Shorthand method, that wraps the is_state_safe method.

        :param state: A state of the grid2op environment do be judged.
        :return: A decision, whether the given state is safe or not.
        """
        return self.is_state_safe(state=state)

    def _get_expected_outage_event(self, state: CompleteObservation) -> Contingency:
        """Get the expected outage event (One event that might contain multiple grid elements). This is currently only
        implemented for maintenance.

        :param state: The current state.
        :return: The outage event as a contingency type.
        """
        expected_outage_list = list()
        for idx in np.where((0 < state.time_next_maintenance) *
                            (state.time_next_maintenance <= self._expected_outage_lookahead))[0]:
            expected_outage_list.append(GridElement(GridElementType.line, idx))
        return ExpectedContingency(expected_outage_list)

    def _get_expected_post_contingency_state(self, state: CompleteObservation, contingency: Contingency) \
            -> Tuple[Optional[CompleteObservation], bool, bool]:
        """Get the expected post contingency state.

        :param state: The current state of the wrapped environment.
        :param contingency: The contingency to simulated. (The Gird elements, we want to simulate an outage of).

        :return: The expected (simulated) state after the outage of the contingency occurred, a bool indicating
                 whether a blackout occurred during simulation and another boolean indicating whether this contingency
                 should be skipped (in case line is already offline).
        """
        action = contingency.to_action_dict(state)

        if sum(map(len, action.values())) == 0:
            # In case no action is currently valid skip this contingency without simulation.
            return None, False, True

        playable_action = self._wrapped_env_action_space(action)
        # Simulate outage action with updated parameters.
        obs, _, done, info = grid2op_contingency_analysis_simulate(state, playable_action,
                                                                   ignore_hard_overflow=False, use_forecast=False)

        if info['is_illegal'] or info['is_ambiguous']:
            # Illegal or ambiguous action should not occur.
            raise ValueError(
                BColors.format_colored(f'The contingency action is illegal or ambiguous: {info}, {action}, '
                                       f'{playable_action}', BColors.WARNING))

        diverging_power_flow = False
        if done or len(info['exception']) > 0:
            logger.info(f'The contingency action results in a done: {info["exception"]}, {action}, '
                        f'{playable_action}')
            diverging_power_flow = True

        return obs, diverging_power_flow, False

    def _validate_grid(self, state: CompleteObservation) -> Violations:
        """Analyse the grid based on the current maze state (the complete observation from the grid2op wrapped env).

        :param state: A state of the grid2op environment do be judged.
        :return: The violations object holding all violation occurring in the current state.
        """
        # Init violations dict
        violations = Violations()
        if np.all(~state.line_status):
            logger.info(f'All lines offline --> Env is done! No need to validate grid.')
            return violations

        # Run state test
        list_of_violations = list()
        for state_test in self._state_tests:
            list_of_violations.extend(state_test(state))
        violations.set_violations(None, list_of_violations)

        if len(violations) > 0 and not self._calculate_post_contingency_if_state_is_not_safe:
            return violations

        # Process expected contingency event. This means checking if there is an expected maintenance, and simulating
        #   the corresponding outage in the grid.
        expected_contingency = self._get_expected_outage_event(state)
        if len(expected_contingency) > 0:
            post_contingency_state, is_blackout, skip_contingency = \
                self._get_expected_post_contingency_state(state, expected_contingency)
            if skip_contingency:
                logger.warning(f'skipping teh contingency: {expected_contingency} because disabling the lines is not a '
                               f'valid action at this point. offline lines: {np.where(~state.line_status)[0]}')
            else:
                list_of_violations = list()
                if is_blackout:
                    list_of_violations.append(DivergingPowerFlowViolation(grid_element=None,
                                                                          violation_level=ViolationLevel.critical))
                else:
                    for state_test in self._post_expected_outage_state_tests:
                        list_of_violations.extend(state_test(post_contingency_state))
                violations.set_violations(expected_contingency, list_of_violations)

        if len(violations) > 0 and not self._calculate_post_contingency_if_state_is_not_safe:
            return violations

        return violations

    def validate_grid(self, state: CompleteObservation) -> Tuple[Violations, Dict[ViolationLevel, int], bool]:
        """Compute if necessary and return all violations for each of the contingencies.

        :return: A dictionary of all violations. Each key in the returned dictionary is a contingency from the
                 contingency list while corresponding value is a sorted list of violations that occur when an outage
                 of the contingency is simulated (with the noop action). If the key 'None' is present as a key in the
                 dictionary, the corresponding list of violations where evaluated on the current state.
                 Additionally, the count of violations as well as the bool indicated whether this is a safe state is
                 also returned.
        """
        # compute current state hash
        env_done = np.all(~state.line_status)
        if env_done:
            self._latest_is_safe_state = False
            self._latest_violations = Violations()
            self._latest_violation_counts = self._latest_violations.count_violations()
        else:
            # compute current state hash
            state_hash = self._compute_state_hash(state)

            # update critical state hash and status if required
            if state_hash != self._latest_state_hash:
                self._latest_violations = self._validate_grid(state)
                self._latest_violation_counts = self._latest_violations.count_violations()
                self._latest_is_safe_state = self._latest_violation_counts[ViolationLevel.severe] == 0 and \
                                             self._latest_violation_counts[ViolationLevel.critical] == 0 and \
                                             self._latest_violation_counts[ViolationLevel.risky] == 0 and \
                                             not np.all(~state.line_status)
                self._latest_state_hash = state_hash

        return self._latest_violations, self._latest_violation_counts, self._latest_is_safe_state

    def is_state_safe(self, state: CompleteObservation) -> bool:
        """Decide whether the given state represents a safe situation or not.

        :param state: A state of the grid2op environment do be judged.
        :return: A decision, whether the given state is safe or not.
        """
        return self.validate_grid(state)[2]

    def get_violations_counts(self, state: CompleteObservation) -> Dict[ViolationLevel, int]:
        """Compute if necessary and return the counts of all violation for each of the contingencies.

        :return: A dictionary for all violations levels. Each key in the returned dictionary is a violation level and
                 the corresponding integer represents the count of violations of the level occurring in the current
                 state.
        """
        return self.validate_grid(state)[1]

    def get_violations(self, state: CompleteObservation) -> Violations:
        """Retrieve the violations on the current state.

        :param state: The current grid state.
        :return: A dictionary of all violations. Each key in the returned dictionary is a contingency from the
                 contingency list while corresponding value is a sorted list of violations that occur when an outage
                 of the contingency is simulated (with the noop action). If the key 'None' is present as a key in the
                 dictionary, the corresponding list of violations where evaluated on the current state.
        """
        return self.validate_grid(state)[0]

    def get_observation_for_state(self, state: CompleteObservation) -> Dict[str, np.ndarray]:
        """Update the given observation by adding the critical state observation if applicable.

        :param state: The current grid state.
        :return: The updated observation with added critical state if applicable.
        """
        _, violation_counts, is_safe_state = self.validate_grid(state)
        observation = {
            'is_safe_state': np.asarray([is_safe_state], dtype=np.float32),
            'n_violations_severe': np.asarray([violation_counts[ViolationLevel.severe]], dtype=np.float32),
            'n_violations_critical': np.asarray([violation_counts[ViolationLevel.critical]], dtype=np.float32)
        }
        return observation

    def clone_from(self, grid_state_observer: 'GridStateObserver') -> None:
        """Reset the critical state observer to the state of the provided critical state observer.

        :param grid_state_observer: The critical state observer to clone from.
        """
        self._post_expected_outage_state_tests = grid_state_observer._post_expected_outage_state_tests
        self._expected_outage_lookahead = grid_state_observer._expected_outage_lookahead
        self._wrapped_env_action_space = grid_state_observer._wrapped_env_action_space

        self._latest_state_hash = grid_state_observer._latest_state_hash
        self._latest_is_safe_state = grid_state_observer._latest_is_safe_state
        self._latest_violations = copy.deepcopy(grid_state_observer._latest_violations)
        self._latest_violation_counts = copy.deepcopy(grid_state_observer._latest_violation_counts)

    @classmethod
    def _compute_state_hash(cls, state: CompleteObservation) -> str:
        """Returns a hash of the current environment.

        :return: The hash value.
        """
        return str(state.get_time_stamp()) + str(hash(state.rho.tostring()))

    def str_rep_of_latest_violations(self) -> str:
        """Return a string representation of the current violations."""
        return str(self._latest_violations)
