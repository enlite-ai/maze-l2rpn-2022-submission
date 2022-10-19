"""L2RPN rendering classes."""
from typing import Optional

import grid2op
from grid2op.Observation import CompleteObservation
from maze.core.annotations import override
from maze.core.env.maze_action import MazeActionType
from maze.core.log_events.step_event_log import StepEventLog
from maze.core.rendering.renderer import Renderer

from maze_l2rpn.env.custom_plot_matplotlib import CustomPlotMatplot


class L2RPNRenderer(Renderer):
    """Loop-compatible wrapper around the Grid2Op renderer."""

    def __init__(self, observation_space: grid2op.Observation.ObservationSpace):
        self.plotter = CustomPlotMatplot(observation_space)

    @override(Renderer)
    def render(self, maze_state: CompleteObservation, maze_action: Optional[MazeActionType], events: StepEventLog,
               **kwargs) -> None:
        """Render the current state using the Grid2Op matplotlib renderer."""
        self.plotter.plot_obs(maze_state, **kwargs)

    def reset_values(self):
        """Reset the historic values"""
        self.plotter.reset_values()
