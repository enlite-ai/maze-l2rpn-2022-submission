"""
DO NOT REMOVE
"""
import warnings
import grid2op

SUPPORTED_GRID2OP_VERSION = ['1.7.2']

warnings.filterwarnings("ignore")

APPlY_GRID2OP_BUGFIX = False
APPLY_RECONNECTION_BUGFIX_TO_SIMULATION = True
APPlY_GRID2OP_BUGFIX_TO_SIMULATION = False
APPLY_GRID2OP_BUGFIX_TO_ACTION = False

# Either fix the bug in the env directly (this is however not possible on the deployment server) or fix it in the
# simulation.

assert grid2op.__version__ in SUPPORTED_GRID2OP_VERSION, \
    f'The installed grid2op version ({grid2op.__version__}) is currently not supported. ' \
    f'Supported versions: {SUPPORTED_GRID2OP_VERSION}'
