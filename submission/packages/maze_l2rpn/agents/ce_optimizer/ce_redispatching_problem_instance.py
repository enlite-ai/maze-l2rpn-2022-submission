"""Cross-entropy interfaces for the redispatching optimizer."""
from typing import Optional, Tuple, List

import numpy as np
from grid2op.Action import PlayableAction, ActionSpace
from grid2op.Observation import CompleteObservation

from maze_l2rpn.agents.ce_optimizer.ce_optimizer import CEProblemInstance
from maze_l2rpn.simulate.grid2op_ca_simulation import grid2op_contingency_analysis_simulate
from maze.core.annotations import override


class CERedispatchingProblemInstance(CEProblemInstance):
    """This object is used to model the optimal redispatching problem instance for a specific env step.

    :param problem: The associated problem definition.
    :param action_space: grid2op ActionSpace, used to construct PlayableAction instances.
    :param observation: Observation to optimize.
    :param line_contingencies: List of line ids to simulate post-contingency states.
    :param lines_to_relieve: List of line ids that should be relieved as much as possible.
    """

    def __init__(self, problem: "CERedispatchingProblem", action_space: ActionSpace,
                 observation: CompleteObservation, joint_action: Optional[PlayableAction],
                 line_contingencies: List[int], lines_to_relieve: List[int]):
        redisp_indices = np.where((observation.gen_max_ramp_up > 0) | (observation.gen_max_ramp_down < 0))[0]
        curtail_indices = np.where(observation.gen_renewable)[0]

        # construct the decision space according to the problem definition
        # (redispatching and/or storage and/or curtailment)
        u_min = []
        u_max = []
        u_init_loc = []
        u_init_scale = []

        # construct Python array slice objects for convenient projection of the solution vector u to its sub-action
        redisp_slice: Optional[slice] = None
        storage_slice: Optional[slice] = None
        curtail_slice: Optional[slice] = None

        if problem.redispatch:
            offset = sum(len(a) for a in u_min)
            redisp_slice = slice(offset, offset + len(redisp_indices))

            # Unfortunately in Grid2Op intended redispatching and actual redispatching often differ by a large margin!
            # Background information:
            # * `target_dispatch` is the redispatching set by the agent over the previous env interactions
            # * `actual_dispatch` is what grid2op applied in the environment after
            #   1. normalizing `target_dispatch` to
            #      sum(actual_dispatch) = sum(storage power flow) + sum(effective curtailment)
            #   2. enforcing generator ramping constraints, which need to be checked towards to the new generator
            #      dispatch of the current timestamp, which reflects fluctuations in demand and renewable generation
            #      not known in the previous env step.
            # * ramping constraints are (unnecessarily) applied to any change in `target_dispatch`, which the agent
            #   controls via `redispatch` action.

            # Avoid that target_dispatch and actual_dispatch drift apart by narrowing the ramp_up and ramp_down limits.
            # Idea: Forbid generator elements in target_dispatch to move further away from actual_dispatch than
            #       the interval [actual_dispatch - ramp_down, actual_dispatch + ramp_up] allows.
            drift_correction = np.clip(observation.target_dispatch - observation.actual_dispatch,
                                       -observation.gen_max_ramp_down, observation.gen_max_ramp_up)
            max_ramp_down = observation.gen_max_ramp_down - np.maximum(0, -drift_correction)
            max_ramp_up = observation.gen_max_ramp_up - np.maximum(0, drift_correction)
            # fine-tune ramp-up and ramp-down ranges to lie within gen_pmin and gen_pmax
            # in grid2op 1.7.1 gen_p is sometimes negative!
            gen_p = np.maximum(observation.gen_p, 0)
            max_ramp_down = np.minimum(gen_p - observation.gen_pmin, max_ramp_down)
            max_ramp_up = np.minimum(observation.gen_pmax - gen_p, max_ramp_up)

            # init CE parameters
            redispatch_min = -max_ramp_down[redisp_indices]
            redispatch_max = max_ramp_up[redisp_indices]

            u_min.append(redispatch_min)
            u_max.append(redispatch_max)
            u_init_loc.append(np.zeros_like(redispatch_min))
            # redispatching actions are robust, initial samples can cover a large part of the decision range
            u_init_scale.append((redispatch_max - redispatch_min) / 4.0)

        if problem.storage:
            offset = sum(len(a) for a in u_min)
            storage_slice = slice(offset, offset + len(observation.storage_max_p_absorb))

            # consider storage charge limits
            hours = observation.delta_time / 60.0
            energy_to_max = (observation.storage_Emax - observation.storage_charge
                             ) / observation.storage_charging_efficiency
            energy_to_min = (observation.storage_charge - observation.storage_Emin
                             ) * observation.storage_discharging_efficiency
            max_power_charging = energy_to_max / hours
            max_power_production = energy_to_min / hours
            storage_power_max = np.minimum(max_power_charging, observation.storage_max_p_prod)
            storage_power_min = -np.minimum(max_power_production, observation.storage_max_p_absorb)

            u_min.append(storage_power_min)
            u_max.append(storage_power_max)
            u_init_loc.append(np.clip(observation.storage_power, storage_power_min, storage_power_max))
            # storage actions are robust, initial samples can cover a large part of the decision range
            u_init_scale.append((observation.storage_max_p_prod + observation.storage_max_p_absorb) / 4.0)

        if problem.curtail:
            offset = sum(len(a) for a in u_min)
            curtail_slice = slice(offset, offset + len(curtail_indices))
            u_min.append(np.zeros(len(curtail_indices)))
            u_max.append(np.ones(len(curtail_indices)))
            # center curtailment around effective threshold
            u_init_loc.append((observation.gen_p / observation.gen_pmax)[curtail_indices])
            # curtailment ranges are quite large, we need to be more restrictive with the sampling range
            u_init_scale.append(np.full(len(curtail_indices), 0.1))

        self.problem = problem
        self.maze_state = observation
        self.joint_action = joint_action
        self.line_contingencies = line_contingencies
        self.lines_to_relieve = lines_to_relieve
        self.grid2op_action_space = action_space

        self.redisp_indices = redisp_indices
        self.curtail_indices = curtail_indices

        self.redisp_slice = redisp_slice
        self.storage_slice = storage_slice
        self.curtail_slice = curtail_slice

        self.u_init_loc = np.concatenate(u_init_loc)
        self.u_init_scale = np.concatenate(u_init_scale)
        self.u_min = np.concatenate(u_min)
        self.u_max = np.concatenate(u_max)

        # take care of rounding errors
        self.u_max = np.maximum(self.u_max, self.u_min)

    @override(CEProblemInstance)
    def get_decision_variables(self) -> Tuple:
        """Implementation of the CEProblemInstance interface - returns the decision space."""
        return self.u_init_loc, self.u_init_scale, self.u_min, self.u_max

    @override(CEProblemInstance)
    def cost_fn(self, u: np.ndarray, debug=False) -> float:
        """Simulate the redispatching/storage/curtailment action and extract a cost value according to the following
        scheme:
        * `+inf` if the base case fails
        * `1000 + sum(all rho above 100%)` sum up the rho values of all critical lines (above 100%)
        * `100 + max rho of base state` if max rho is above the threshold
        * 10 + max rho of the contingency line, if any post contingency simulation failed
        * `sum(post contingency max rho)` if any contingency violation is detected
        * `operation score / 1e6` if no violations
        * 0 if the env is done with success in the next step
        """
        redispatching_action = self.to_action(u, debug=debug)
        base_case_action = redispatching_action if self.joint_action is None else redispatching_action + self.joint_action

        try:
            base_case_state, base_case_reward, base_case_done, base_case_info = \
                grid2op_contingency_analysis_simulate(state=self.maze_state, playable_action=base_case_action,
                                                      use_forecast=True, ignore_hard_overflow=True)
        except AssertionError as e:
            return np.inf

        if base_case_done:
            base_case_failed = "exception" in base_case_info and len(base_case_info["exception"]) > 0
            if base_case_failed:
                return np.inf

        base_state_rho = base_case_state.rho.max()
        if base_state_rho >= 1.0:
            # lower the load on ALL line loads above 100%
            return 1000 + sum(base_case_state.rho[base_case_state.rho >= 1.0])
        if base_state_rho >= self.problem.max_rho_base:
            return 100 + base_state_rho
        if base_case_done:
            return 0

        risk = []

        # add `lines_to_relieve` to the optimization goal
        for line_id in self.lines_to_relieve:
            risk.append(base_case_state.rho[line_id])

        for contingency in self.line_contingencies:
            contingency_action_dict = contingency.to_action_dict(self.maze_state)

            playable_contingency_action = self.grid2op_action_space(contingency_action_dict) + base_case_action
            assert isinstance(playable_contingency_action, PlayableAction)

            obs, _, done, info = grid2op_contingency_analysis_simulate(
                self.maze_state, playable_contingency_action,
                ignore_hard_overflow=True,
                use_forecast=True)

            if done:
                risk.append(10 + sum(base_case_state.rho[c.idx] for c in contingency))

            post_contingency_max_rho = obs.rho.max()
            if post_contingency_max_rho >= 1.9:
                risk.append(obs.rho.max())

        # in case no contingency is left, try to further decrease max rho
        if not len(risk):
            return base_state_rho

        return sum(risk)

    def forecast_power_change(self, storage: np.ndarray, forecasted_effective_curtailment: np.ndarray) -> float:
        """Get the expected power change of the dispatchable generators.

        :param storage: The storage action
                        (power setpoint, note that in grid2op positive is charging, negative power generation)
        :param forecasted_effective_curtailment: Estimated effective curtailment in MW in the next timestep.
        """
        # We need to replicate the exact same logic as in grid2op BaseEnv._detect_infeasible_dispatch.
        # > sum_move = (
        # >     np.sum(incr_in_chronics) + self._amount_storage - self._sum_curtailment_mw
        # > )

        # The calculation of incr_in_chronics involves a variable not present in the env state.
        # from BaseEnv._compute_dispatch_vect
        # > incr_in_chronics = new_p - (
        # >     self._gen_activeprod_t_redisp - self._actual_dispatch
        # > )
        gen_activeprod_t_redisp_init = self.maze_state._obs_env.gen_activeprod_t_redisp_init
        chronics_increment = (sum(self.maze_state.get_forecasted_inj(1)[0][self.maze_state.gen_redispatchable]) -
                              sum(gen_activeprod_t_redisp_init[self.maze_state.gen_redispatchable]))

        # this corresponds to `sum_move` in the grid2op snippet above
        forecasted_move = (
                chronics_increment + sum(self.maze_state.actual_dispatch) +
                sum(storage) - sum(self.maze_state.storage_power) +
                sum(forecasted_effective_curtailment) - sum(self.maze_state.curtailment_mw)
        )

        return forecasted_move

    def truncate_storage_and_curtailment(self, forecasted_move: float,
                                         max_up: float, max_down: float,
                                         storage: np.ndarray,
                                         curtailment: np.ndarray,
                                         forecasted_p_before_curtailment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Given the forecasted power change exceeds the limits, this method truncates the storage power and
        redispatching so that the max_up, max_down limitations are met.

        :param forecasted_move: Expected power change.
        :param max_up: Ramping constraint, total redispatching in MW.
        :param max_down: Ramping constraint, total redispatching in MW.
        :param storage: Planned storage power action.
        :param curtailment: Planned curtailment action.
        :param forecasted_p_before_curtailment: Forecasted available renewable generation in MW.

        :returns The new action tuple (storage, curtailment)
        """
        forecasted_p_renewable = np.minimum(forecasted_p_before_curtailment,
                                            self.maze_state.gen_pmax[self.curtail_indices] * curtailment)
        forecasted_effective_curtailment = forecasted_p_before_curtailment - forecasted_p_renewable

        if forecasted_move < 0:
            # too much additional power, we need to reduce storage discharge or increase curtailment
            # (otherwise we might not be able to ramp down the redispatchable generators fast enough)
            total_elasticity = sum(storage[storage < 0]) + sum(forecasted_p_renewable)
            f = (-forecasted_move - max_down) / total_elasticity
            f = np.clip(f, 0, 1)
            storage[storage < 0] *= 1 - f
            desired_effective_curtailment = forecasted_p_before_curtailment - forecasted_p_renewable * (1 - f)
        else:
            total_elasticity = sum(-storage[storage > 0]) + sum(forecasted_effective_curtailment)
            f = (forecasted_move - max_up) / total_elasticity
            f = np.clip(f, 0, 1)
            storage[storage > 0] *= 1 - f
            desired_effective_curtailment = forecasted_effective_curtailment * (1 - f)

        # infer the curtailment action (range 0...1) from the given effective curtailment (in MW)
        curtailment = (forecasted_p_before_curtailment - desired_effective_curtailment) / self.maze_state.gen_pmax[
            self.curtail_indices]

        return storage, curtailment

    def limit_power_change(self, storage: np.ndarray, curtailment: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Changes in renewable generation, storage power and curtailment must be absorbed by the redispatchable
        generators, which are limited by generator ramps. This method pushes the storage and curtailment actions down
        to stay within `max_forecasted_power_change`.

        :param storage: The storage action
                        (power setpoint, note that in grid2op positive is charging, negative power generation)
        :param curtailment: Curtailment action in the grid2op sense (actually it is the renewable generation limit,
                            0=maximum curtailment, 1=not limited, no curtailment)
        :return The tuple (curtailment, storage, forecasted_effective_curtailment), `curtailment` and `storage`
                is the processed action after the limitation, `forecasted_effective_curtailment` is the forecasted
                result of the returned curtailment action.
        """
        # forecasting is provided by the injections
        forecasted_p_before_curtailment = self.maze_state.get_forecasted_inj()[0][self.curtail_indices]
        forecasted_p_renewable = np.minimum(forecasted_p_before_curtailment,
                                            self.maze_state.gen_pmax[self.curtail_indices] * curtailment)
        forecasted_effective_curtailment = forecasted_p_before_curtailment - forecasted_p_renewable

        # forecast the power change
        forecasted_move = self.forecast_power_change(storage=storage,
                                                     forecasted_effective_curtailment=forecasted_effective_curtailment)

        # we can't make use of the full margin, as there is uncertainty in the forecast
        max_up = sum(self.maze_state.gen_margin_up) - self.problem.max_forecasting_error
        max_down = sum(self.maze_state.gen_margin_down) - self.problem.max_forecasting_error

        # no need to limit the change
        if -max_down <= forecasted_move <= max_up:
            return curtailment, storage, forecasted_effective_curtailment

        storage, curtailment = self.truncate_storage_and_curtailment(
            forecasted_move=forecasted_move,
            max_down=max_down, max_up=max_up,
            storage=storage, curtailment=curtailment,
            forecasted_p_before_curtailment=forecasted_p_before_curtailment)

        # update effective curtailment forecast
        forecasted_p_renewable = np.minimum(forecasted_p_before_curtailment,
                                            self.maze_state.gen_pmax[self.curtail_indices] * curtailment)
        forecasted_effective_curtailment = forecasted_p_before_curtailment - forecasted_p_renewable

        # handle rounding errors
        curtailment = np.clip(curtailment, 0, 1)
        storage = np.clip(storage, -self.maze_state.storage_max_p_prod, self.maze_state.storage_max_p_absorb)

        return curtailment, storage, forecasted_effective_curtailment

    def balance_redispatch(self,
                           redispatch: np.ndarray, redispatch_min: np.ndarray, redispatch_max: np.ndarray,
                           effective_curtailment: np.ndarray, storage: np.ndarray) -> np.ndarray:
        """The sum of all redispatching actions must be in balance with the storage and curtailed power.
        This method modifies the redispatching vector by distributing the balancing error, respecting the
        min/max limits dictated by the generator ramps.

        :param redispatch: The input redispatching vector to be balanced.
        :param redispatch_min: Defines the minimum values allowed in `redispatch`.
        :param redispatch_max: Defines the maximum values allowed in `redispatch`.
        :param effective_curtailment: The expected curtailed power in MW.
        :param storage: Storage power setpoints.

        :returns The normalized vector.
        """
        redispatch = redispatch.copy()
        balance_error = (sum(redispatch) + sum(self.maze_state.target_dispatch) +
                         sum(-storage) + sum(-effective_curtailment))
        if balance_error > 0:
            margin = redispatch - redispatch_min
            remaining = np.ones_like(redispatch, dtype=bool)
            while True:
                delta = balance_error / np.sum(remaining)
                saturated = (margin < delta) & remaining
                if np.sum(saturated):
                    redispatch[saturated] = redispatch_min[saturated]
                    balance_error -= sum(margin[saturated])
                    remaining = remaining & ~saturated
                else:
                    redispatch[remaining] -= delta
                    break
        if balance_error < 0:
            balance_error = -balance_error
            margin = redispatch_max - redispatch
            remaining = np.ones_like(redispatch, dtype=bool)
            while True:
                delta = balance_error / np.sum(remaining)
                saturated = (margin < delta) & remaining
                if np.sum(saturated):
                    redispatch[saturated] = redispatch_max[saturated]
                    balance_error -= sum(margin[saturated])
                    remaining = remaining & ~saturated
                else:
                    redispatch[remaining] += delta
                    break

        # take care of rounding errors
        return np.clip(redispatch, redispatch_min, redispatch_max)

    @override(CEProblemInstance)
    def to_action(self, u: np.ndarray, debug=False) -> PlayableAction:
        """Create a grid2op PlayableAction from the decision vector"""
        action_dict = dict()
        storage = np.zeros_like(self.maze_state.storage_power)
        curtailment = np.ones(len(self.curtail_indices))
        if self.storage_slice:
            storage = u[self.storage_slice]
        if self.curtail_slice:
            curtailment = u[self.curtail_slice]

        # keep the power changes within a certain range to avoid prematurely terminating the env by
        # an InvalidRedispatching exception
        curtailment, storage, effective_curtailment = self.limit_power_change(storage=storage, curtailment=curtailment)

        action_dict["set_storage"] = storage
        action_dict["curtail"] = [(self.curtail_indices[i], curtailment[i]) for i in
                                  range(len(self.curtail_indices))]

        if self.redisp_slice:
            redispatch = u[self.redisp_slice]
            redispatch_min = self.u_min[self.redisp_slice]
            redispatch_max = self.u_max[self.redisp_slice]

            # balance the redispatching vector, taking into account the storage and curtailment action
            redispatch = self.balance_redispatch(
                redispatch=redispatch, redispatch_min=redispatch_min, redispatch_max=redispatch_max,
                effective_curtailment=effective_curtailment, storage=storage)

            all_gen = np.zeros(self.maze_state.n_gen)
            all_gen[self.redisp_indices] = redispatch

            action_dict["redispatch"] = all_gen

        action = self.grid2op_action_space(action_dict)
        assert isinstance(action, PlayableAction)
        return action
