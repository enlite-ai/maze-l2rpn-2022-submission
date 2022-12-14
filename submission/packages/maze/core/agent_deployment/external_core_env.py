"""Acts as a core env in an Agent Deployment setting."""
from queue import Queue
from threading import Event
from typing import Tuple, Any, Dict, Union, Iterable, Optional

import numpy as np

from maze.core.annotations import override
from maze.core.env.core_env import CoreEnv
from maze.core.env.environment_context import EnvironmentContext
from maze.core.env.event_env_mixin import EventEnvMixin
from maze.core.env.maze_action import MazeActionType
from maze.core.env.maze_state import MazeStateType
from maze.core.env.structured_env import StepKeyType, ActorID
from maze.core.events.event_record import EventRecord
from maze.core.log_events.kpi_calculator import KpiCalculator
from maze.core.rendering.renderer import Renderer


class ExternalCoreEnv(CoreEnv):
    """Acts as a CoreEnv in the env stack in agent deployment scenario.

    Designed to be run on a separate thread, alongside the agent deployment running on the main thread.

    Hence, the control flow is: External env (like a Unity env) controlling the agent deployment object,
    which in turn controls this external core env, which controls the execution of rollout loop by suspending it
    until the next state is available from the agent deployment object.

    Wrappers of this env and the agents acting on top of it see it as ordinary CoreEnv, but no actual
    logic happens here -- instead, states and associated info are obtained from the agent deployment
    running on the main thread, and executions produced by the agents are passed back to the agent deployment.

    During the step function, the execution of this thread is suspended while waiting for the next state
    from the agent deployment.

    :param state_queue: Queue this core env uses to get states from agent deployment object
    :param maze_action_queue: Queue this core env uses to pass executions back to agent deployment object
    :param rollout_done_event: Set by the agent deployment object. Used for detection of the end of rollout period.
    :param renderer: If available, what renderer should be associated with the state data (for rendering, plus
                     to be serialized with trajectory data)
    """

    def __init__(self,
                 context: EnvironmentContext,
                 state_queue: Queue,
                 maze_action_queue: Queue,
                 rollout_done_event: Event,
                 renderer: Optional[Renderer] = None,
                 kpi_calculator: Optional[KpiCalculator] = None):
        super().__init__()
        self.context = context
        self.state_queue = state_queue
        self.maze_action_queue = maze_action_queue
        self.rollout_done_event = rollout_done_event
        self.renderer = renderer
        self.kpi_calculator = kpi_calculator

        self.last_maze_state = None
        self._actor_id = (0, 0)
        self._is_actor_done = False

    # --- Step & reset: The core of ExternalCoreEnv functionality ---

    @override(CoreEnv)
    def reset(self) -> MazeStateType:
        """Reset is expected to be run twice -- at the beginning and end of external env rollout.

        At the beginning, thread execution is suspended until the initial state is available.

        At the end of the rollout, just the last state is returned, as there the reset serves the only purpose
        of notifying the wrappers to do their processing of the previous episode. (Also, no more states are
        available from the external env at this point.
        """
        # If the external env has been declared done, just return the last state again (as no more states are available)
        if not self.rollout_done_event.is_set():
            self.last_maze_state, _, _, _, events = self.state_queue.get()
            self._replay_events(events)

        return self.last_maze_state

    @override(CoreEnv)
    def step(self, maze_action: MazeActionType) -> Tuple[
        MazeStateType, Union[float, np.ndarray, Any], bool, Dict[Any, Any]]:
        """Relays the execution back to the agent deployment. Then suspends thread execution until
        the next state is provided by agent deployment."""
        self.maze_action_queue.put(maze_action)

        # Here, thread execution is suspended until the next state object is put in the queue by AIW.
        # (This happens when the external env controlling the AIW queries it for the next execution.)

        state, reward, done, info, events = self.state_queue.get()
        self.last_maze_state = state
        self._replay_events(events)

        # Increment step and clear events - structured core environments are not supported
        self.context.increment_env_step()

        return state, reward, done, info

    # --- Structured env methods and setters ---

    def set_actor_id(self, new_value: ActorID):
        """Hook for the agent deployment to set actor_id before querying execution."""
        self._actor_id = new_value

    @override(CoreEnv)
    def actor_id(self) -> ActorID:
        """Current actor ID set by the agent deployment."""
        return self._actor_id

    def set_is_actor_done(self, new_value: bool):
        """Hook for the agent deployment to set the actor_done flag before querying execution."""
        self._is_actor_done = new_value

    @override(CoreEnv)
    def is_actor_done(self) -> bool:
        """Whether last actor is done, as set by the agent deployment."""
        return self._is_actor_done

    @property
    @override(CoreEnv)
    def agent_counts_dict(self) -> Dict[StepKeyType, int]:
        """Agent counts are not known and not needed, as this env is not used for training or any other setup."""
        return {0: -1}

    # --- The rest of core env methods ---

    @override(CoreEnv)
    def get_maze_state(self) -> MazeStateType:
        """Return the last state obtained from the external env through agent deployment."""
        return self.last_maze_state

    @override(CoreEnv)
    def get_renderer(self) -> Optional[Renderer]:
        """Renderer provided by the agent deployment. Might be None if not available. """
        return self.renderer

    @override(EventEnvMixin)
    def get_step_events(self) -> Iterable[EventRecord]:
        """Get all events recorded in the current step from the EventService."""
        return self.context.event_service.iterate_event_records()

    @override(CoreEnv)
    def get_kpi_calculator(self) -> Optional[KpiCalculator]:
        """KPI calculator provided by the agent deployment. Might be None if not available. """
        return self.kpi_calculator

    @override(CoreEnv)
    def get_serializable_components(self) -> Dict[str, Any]:
        """No components available."""
        return {}

    @override(CoreEnv)
    def seed(self, seed: int) -> None:
        """No seed required -- all operation handled by external env."""
        pass

    @override(CoreEnv)
    def close(self) -> None:
        """No cleanup required."""
        pass

    # -- External core env helpers --

    def _replay_events(self, events: Optional[Iterable[EventRecord]]) -> None:
        """Notify the event service about each of the events passed in from outside."""
        if events:
            for event in events:
                self.context.event_service.notify_event(event)
