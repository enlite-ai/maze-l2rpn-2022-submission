"""Contains Maze-RL submission config."""
import logging

from maze_l2rpn.agents.simulation_search_submission import SimulationSearchSubmissionAgent

EXPERIMENT_PATH = 'experiment_data'

logger = logging.getLogger('MAZE_SUBMISSION')
logger.setLevel(logging.INFO)


class SubmissionParams:
    """Maze-RL submission config."""

    # SimulationSearchSubmissionAgent
    N_CANDIDATES = 25
    EARLY_STOPPING_MAX_RHO = 0.95
    N_CANDIDATES_UNSAFE = 150
    MIN_CANDIDATES = 5

    # Recovery
    RECOVERY_MAX_RHO = 0.97
    RECOVERY_SKIP_N_STEPS_AFTER_RECOVERY = 10
    RECOVERY_TOPO_VEC = None
    WITH_RECOVERY = True
    RECOVERY_RHO_SAFE = 0.9
    CHECK_FULL_RECOVERY = True

    # If true agents always returns NOOP
    ONLY_NOOP = False

    # select policy to use
    maze_policy = SimulationSearchSubmissionAgent

    # Perform assertions
    ASSERT_ENV_DYNAMICS = False

    LOGGING_LEVEL = 'info'
