"""Contains a short script that loads up all the packages on the submission server
"""

import sys
import os

# this tool is separated from the project resources, make the modules of the script directory available
local_packages = os.path.join(os.path.dirname(__file__), "packages")
sys.path.insert(0, local_packages)
