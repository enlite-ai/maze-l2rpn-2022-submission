"""File containing a validation method for the output of the grid2op step function or simulate function."""
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import grid2op
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation

logger = logging.getLogger('STEP OUTPUT VALIDATION')
logger.setLevel(logging.WARNING)


def grid2op_step_output_validation(state: CompleteObservation, reward: float, done: bool,
                                   info: Dict[str, Any], playable_action: Optional[PlayableAction],
                                   seed_info: Any, was_simulated: bool) -> Tuple[CompleteObservation, float, bool,
                                                                                 Dict[str, Any]]:
    """Check the result of the grid2op step or grid2op simulate method.

    :param state: The resulting state.
    :param reward: The resulting reward.
    :param done: The resulting done.
    :param info: The resulting info dict.
    :param playable_action: The Playable action that was used.
    :param seed_info: The seed info.
    :param was_simulated: If this method was called on simulated output.

    :return: The quadruple of obs, rew, done and info.
    """
    # Check if action was illegal or ambiguous
    if info['is_illegal'] or info['is_ambiguous']:
        txt = 'SIMULATION' if was_simulated else 'ENV STEP'
        txt += f' is {"ambiguous" if info["is_ambiguous"] else "illegal"} due to: {info["exception"]} ' \
               f'-> with action: {playable_action} -- seed info: {seed_info}'
        logger.exception(txt)
        if was_simulated:
            done = True
    # Remove nan values
    state.rho = np.nan_to_num(state.rho, nan=np.inf)

    if done:
        state.line_status = np.zeros_like(state.line_status)

    return state, reward, done, info
