"""Cross-entropy interfaces for the redispatching optimizer."""
from typing import Optional, List

from grid2op.Action import ActionSpace, PlayableAction
from grid2op.Observation import CompleteObservation

from maze_l2rpn.agents.ce_optimizer.ce_optimizer import CEProblem
from maze_l2rpn.agents.ce_optimizer.ce_redispatching_problem_instance import CERedispatchingProblemInstance
from maze_l2rpn.grid_state_observer.grid_state_observer_types import Contingency, GridElement
from maze.core.annotations import override


class CERedispatchingProblem(CEProblem):
    """Implementation of the problem definition interface for redispatching.

    :param redispatch: Add the redispatch action to the search space.
    :param storage: Add storage control to the search space.
    :param curtail: Add curtailment to the search space.
    :param max_forecasting_error: Forecasting of generation power is used to estimate the power changes that need to be
                            absorbed by the redispatchable generators. This value specifies a safety margin in MW.
    :param rho_danger_base: If any line load of the base case exceeds this value, the optimizer is directed to
                            prioritize the reduction of this load over any contingency improvements.
                            (base case=plain simulation of next grid step, without contingency analysis).
    :param contingencies: A list of contingencies to analyse - same definition as used by the GridStateObserver.
    """

    def __init__(self,
                 redispatch: bool,
                 storage: bool,
                 curtail: bool,
                 max_forecasting_error: float,
                 rho_danger_base: float,
                 contingencies: Optional[List[Contingency]]):
        self.redispatch = redispatch
        self.storage = storage
        self.curtail = curtail
        self.max_forecasting_error = max_forecasting_error

        self.max_rho_base = rho_danger_base
        self.contingencies = [Contingency([GridElement.build(ge) for ge in contingency])
                              for contingency in contingencies] if contingencies is not None else []

    @override(CEProblem)
    def create_instance(self, action_space: ActionSpace, observation: CompleteObservation,
                        joint_action: PlayableAction,
                        line_contingencies: List[int], lines_to_relieve: List[int]) -> CERedispatchingProblemInstance:
        """Implementation of the CEProblem interface - returns a problem instances."""
        return CERedispatchingProblemInstance(problem=self, action_space=action_space, observation=observation,
                                              joint_action=joint_action,
                                              line_contingencies=line_contingencies,
                                              lines_to_relieve=lines_to_relieve)
