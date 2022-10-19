"""Critical state observer that also check whether any line has been disconnected."""
from typing import List, Union

from grid2op.Observation import CompleteObservation
from maze_l2rpn.grid_state_observer.grid_state_observer_types import ViolationLevel
from maze_l2rpn.grid_state_observer.state_tests.state_test_interface import StateTestInterface
from maze_l2rpn.grid_state_observer.violations.reward_threshold_violation import RewardThresholdViolation
from maze_l2rpn.grid_state_observer.violations.violation import Violation
from maze.core.annotations import override


class RewardThresholdStateTest(StateTestInterface):
    """A state test that checks if a line has been disconnected by the backend.

    :param violation_level: The violation level the state tests result should correspond to.
    """

    def __init__(self, violation_level: Union[str, ViolationLevel], reward_name: str, reward_threshold: float,
                 check_lower_bound: bool):
        super().__init__(violation_level)
        self.reward_name = reward_name
        self.reward_threshold = reward_threshold
        self.check_lower_bound = check_lower_bound

    @override(StateTestInterface)
    def __call__(self, state: CompleteObservation) -> List[Violation]:
        """Return a LineDisconnectedViolation for each line that has been disconnected.

        :param state: The state to be checked.
        :return: A list of all violations, that is a list of all line limit violations occurring.
        """
        violations = list()
        if self.check_lower_bound and state.rewards[self.reward_name] < self.reward_threshold:
            violations.append(RewardThresholdViolation(grid_element=None, violation_level=self._violation_level,
                                                       state_reward=state.rewards[self.reward_name],
                                                       reward_threshold=self.reward_threshold,
                                                       reward_name=self.reward_name))
        elif not self.check_lower_bound and state.reward[self.reward_name] >= self.reward_threshold:
            violations.append(RewardThresholdViolation(grid_element=None, violation_level=self._violation_level,
                                                       state_reward=state.rewards[self.reward_name],
                                                       reward_threshold=self.reward_threshold,
                                                       reward_name=self.reward_name))

        return violations

    @override(StateTestInterface)
    def clone_from(self, state_test: 'LineDisconnectedStateTest') -> None:
        """Reset the state test to the state of the provided state test.

        :param state_test: The state test to clone from.
        """
