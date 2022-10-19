"""Cross-entropy optimizer implementation."""
import logging
from abc import ABC
from typing import Tuple, Callable, List

import numpy as np
from grid2op.Action import ActionSpace, PlayableAction
from grid2op.Observation import CompleteObservation
from scipy.stats import norm

from maze.core.env.maze_action import MazeActionType

logger = logging.getLogger('CEPolicy')


class CEProblem(ABC):
    """Specifies an optimization problem for the black-box optimizer."""

    def create_instance(self,
                        action_space: ActionSpace,
                        observation: CompleteObservation,
                        joint_action: PlayableAction,
                        line_contingencies: List[int],
                        lines_to_relieve: List[int]) -> "CEProblemInstance":
        """Constructs an optimization problem instance for the given grid2op observation."""


class CEProblemInstance(ABC):
    """Representing an optimization instance for a specific env step."""

    def cost_fn(self, u: np.ndarray) -> float:
        """Infer the cost (=optimization objective) from the given decision variable.

        :return The cost to be minimized as scalar float.
        """

    def get_decision_variables(self) -> Tuple:
        """Returns the decision space according to the problem definition.

        :return A tuple (u_init_loc, u_init_scale, u_min, u_max)
                * u_init_loc: The initial decision vector for the optimizer.
                * u_init_scale: The variance used by the black-box optimizer to draw the initial population
                * u_min: Element-wise lower bound of the decision vector
                * u_max: Element-wise upper bound of the decision vector
        """

    def to_action(self, u: np.ndarray) -> MazeActionType:
        """Convert the given decision variable vector to an env action."""


class CEOptimizer:
    """Cross entropy optimizer

    :param population_size: Number of simulated samples per optimization epoch.
    :param rho: Fraction of winning samples relative to entire population (usually in the range 0.01 - 0.1).
    :param n_generations: Number of optimization updates.
    :param smoothing: Optionally apply smoothing to the variance estimation
                      (0.0=no smoothing, 1.0=keep initial variance indefinitely)
    """

    def __init__(self,
                 population_size: int,
                 rho: float,
                 n_generations: int,
                 smoothing: float):
        self.rng = np.random.RandomState(None)

        assert n_generations > 0
        assert 0.0 < rho <= 1.0
        assert population_size > 0
        assert 0.0 <= smoothing <= 1.0

        self.population_size = population_size
        self.rho = rho
        self.n_generations = n_generations
        self.smoothing = smoothing

    def optimize(self,
                 u_loc: np.ndarray, u_scale: np.ndarray,
                 u_min: np.ndarray, u_max: np.ndarray,
                 cost_fn: Callable[[np.ndarray], float]) -> np.ndarray:
        """Run the optimization process.

        :param u_loc: Initial mean values. The optimization process is only effective if these values describe a
                      feasible solution (otherwise feasible solutions are searched at random).
        :param u_scale: Initial variance. Hint: Keep the variance at a level that keeps the sampled simulations
                        mostly feasible but still provides enough variance to allow for effective optimization updates.
        :param u_min: Defines the lower limit of the decision variable range
        :param u_max: Defines the upper limit of the decision variable range
        :param cost_fn: A function f(u) mapping a decision variable (=solution) to cost values (=objective).
                        Usually the cost function involves the grid simulation and extracting performance
                        indicators from the resulting env state.

        :returns The best solution found as decision variable vector (in general not optimal)
        """
        u_scale = np.maximum(u_scale, 0)

        # experimental code snippet to compare CE with nelder-mead
        #
        # simplex_size = len(u_loc)+1
        # initial_simplex = self.rng.normal(size=(simplex_size, len(u_loc)), loc=u_loc, scale=u_scale)
        # initial_simplex = initial_simplex.clip(u_min, u_max)
        # result = minimize(cost_fn, initial_simplex[0], method='Nelder-Mead', bounds=zip(u_min, u_max),
        #                   options=dict(disp=True, maxiter=400, initial_simplex=initial_simplex,  adaptive=True))

        # keep track of the best instance so far across all generations
        best_cost_found = np.inf
        best_u_found = None

        for t in range(self.n_generations):
            # sample clipped normal distributions
            u_t = self.rng.normal(size=(self.population_size, len(u_loc)), loc=u_loc, scale=u_scale)
            u_t = u_t.clip(u_min, u_max)

            # involve the cost function for all samples
            cost = np.zeros(self.population_size)
            for i in range(self.population_size):
                cost[i] = cost_fn(u_t[i, :])

            # sort by cost to extract the fittest samples
            n_e = int(self.population_size * self.rho)
            best_idxs = np.argsort(cost)[:n_e]
            best_u = u_t[best_idxs]

            # update the gaussian sample distribution
            u_scale_t_plus_1 = np.zeros(len(u_loc))
            for j in range(len(u_loc)):
                u_loc[j], u_scale_t_plus_1[j] = norm.fit(best_u[:, j])
            # apply smoothing to the u_scale update
            u_scale = u_scale * self.smoothing + u_scale_t_plus_1 * (1 - self.smoothing)

            if cost[best_idxs[0]] <= best_cost_found:
                best_cost_found = cost[best_idxs[0]]
                best_u_found = u_t[best_idxs[0]]

            logger.debug(f"generation:{t},"
                         f"best cost:{cost[best_idxs[0]]},"
                         f"mean cost:{np.mean(cost)},"
                         f"feasible count {np.sum(cost < np.inf)}")
            # np.set_printoptions(suppress=True)
            # print(cost_fn(u_t[best_idxs[0]], debug=True))

        return best_u_found
