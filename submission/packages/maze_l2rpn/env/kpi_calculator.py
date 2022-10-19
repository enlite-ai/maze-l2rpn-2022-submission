"""KPIs for the grid2op environment."""

from typing import Dict, Any

import numpy as np
from maze.core.log_events.episode_event_log import EpisodeEventLog
from maze.core.log_events.kpi_calculator import KpiCalculator

from maze_l2rpn.env.events import RewardEvents


class Grid2OpKpiCalculator(KpiCalculator):
    """KPIs for grid2op environment.
    """

    def calculate_kpis(self, episode_event_log: EpisodeEventLog, last_state: Any) -> Dict[str, float]:
        """Calculates the KPIs at the end of episode."""

        kpis = {}
        for event in episode_event_log.query_events(RewardEvents.other_reward):
            if not event.is_kpi:
                continue

            kpi_name = event.name
            if kpi_name not in kpis:
                kpis[kpi_name] = []

            kpis[kpi_name].append(event.reward)

        # reduce kpi lists to numbers
        for kpi, data in kpis.items():
            kpis[kpi] = float(np.sum(data))

        return kpis
