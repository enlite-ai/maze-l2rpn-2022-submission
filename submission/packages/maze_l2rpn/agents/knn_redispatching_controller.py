"""Redispatching controller utilizing a Torch policy."""
import pickle
from typing import Union, List, Optional, Dict, Any, Tuple

from grid2op.Action import ActionSpace, PlayableAction
from grid2op.Environment.Environment import Environment
from grid2op.Observation import CompleteObservation
from maze.core.annotations import override
from maze.core.env.action_conversion import ActionType
from maze.core.utils.config_utils import SwitchWorkingDirectoryToInput

from maze_l2rpn.agents.ce_optimizer.ce_redispatching_problem import CERedispatchingProblem
from maze_l2rpn.agents.redispatching_controller import RedispatchingController
from maze_l2rpn.env.core_env import L2RPNSeedInfo


class KNNRedispatchingController(RedispatchingController):
    """Redispatching controller utilizing a Torch policy.

    :param input_dir: Path from where to load the hydra config and the knn dump.
    """

    def __init__(self,
                 action_space: ActionSpace,
                 env: Environment,
                 input_dir: str
                 ):
        super().__init__()

        self.action_space = action_space
        self.env = env

        with SwitchWorkingDirectoryToInput(input_dir):
            self.neighbors = pickle.load(open('contingency_redispatches.pkl', 'rb'))

        self.problem = CERedispatchingProblem(redispatch=True, storage=False, curtail=True, max_forecasting_error=35,
                                              rho_danger_base=0.95, contingencies=None)

    @override(RedispatchingController)
    def seed(self, seed: Union[L2RPNSeedInfo, int]) -> None:
        """Seed the policy and conversion env."""

    @override(RedispatchingController)
    def compute_action(self,
                       observation: CompleteObservation,
                       line_contingencies: List[int],
                       lines_to_relieve: List[int],
                       joint_action: Optional[PlayableAction] = None) -> Tuple[ActionType, Dict[str, Any]]:
        """Convert the state to observation, get action from the torch policy, and convert it
        back to playable action."""

        best_u = self.neighbors.predict(observation.rho[None]).flatten()

        problem_instance = self.problem.create_instance(
            self.action_space, observation,
            line_contingencies=[],
            lines_to_relieve=[],
            joint_action=None)
        playable_action = problem_instance.to_action(best_u)

        return playable_action, {}
