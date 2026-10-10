"""Independent deterministic structural model and routing safety controller."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from numbers import Real

import numpy as np
import torch

SCHEMA_VERSION = 1
METRICS = ("throughput_packets", "pdr", "drop_ratio", "latency_proxy_ms", "energy_proxy",
           "dropped_packets", "task_completion", "reward")


class ControllerError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)

    def to_dict(self):
        return {"code": self.code, "message": str(self), "action_executed": self.code=="realization_mismatch"}


def calibration_identity(calibration):
    return hashlib.sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DecisionState:
    demand_packets: float
    queues: tuple[float, float, float]
    split: str
    graph_file: str
    window_start: str
    calibration_id: str
    causal_values: tuple[float, ...] = ()
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class CounterfactualOutcome:
    action: int
    action_name: str
    allocation_fractions: tuple
    post_queue: tuple
    throughput_packets: float
    pdr: float
    drop_ratio: float
    latency_proxy_ms: float
    energy_proxy: float
    dropped_packets: float
    task_completion: float
    reward: float
    intermediates: tuple

    def to_dict(self):
        payload = asdict(self)
        payload["intermediates"] = dict(self.intermediates)
        payload["schema_version"] = SCHEMA_VERSION
        payload["backend"] = "exact_simulator_scm"
        return payload


@dataclass(frozen=True)
class SafetyDecision:
    proposed_action: int
    executed_action: int
    ranking: tuple
    q_values: tuple
    outcomes: tuple
    reasons: tuple
    violation_scores: tuple
    safety_feasible: bool
    status: str
    query_seconds: float

    def to_dict(self):
        result = asdict(self)
        result["outcomes"] = [o.to_dict() for o in self.outcomes]
        result["schema_version"] = SCHEMA_VERSION
        result["q_values_are_probabilities"] = False
        return result


class ActionAwareSCM:
    def __init__(self, calibration: dict, environment: dict):
        self.calibration = json.loads(json.dumps(calibration))
        self.environment = json.loads(json.dumps(environment))
        self.calibration_id = calibration_identity(self.calibration)
        self.actions = self.environment["actions"]
        self.capacity = np.array(calibration["service_capacity"], dtype=float)
        self.buffer = np.array(calibration["buffer_capacity"], dtype=float)
        self.base = np.array(calibration["base_latency_ms"], dtype=float)
        self.idle = np.array(calibration["idle_energy"], dtype=float)
        self.dynamic = np.array(calibration["dynamic_energy"], dtype=float)
        arrays = (self.capacity, self.buffer, self.base, self.idle, self.dynamic)
        if any(a.shape != (3,) or not np.isfinite(a).all() for a in arrays):
            raise ControllerError("invalid_calibration", "Calibration arrays must be finite triples")
        if np.any(self.capacity <= 0) or np.any(self.buffer <= 0) or any(np.any(a < 0) for a in arrays[2:]):
            raise ControllerError("invalid_calibration", "Invalid capacities or coefficients")
        self.budget = float(calibration["latency_budget_ms"])
        self.window = float(calibration["window_seconds"])
        self.max_energy = float((self.idle + self.dynamic).sum())
        if not np.isfinite([self.budget, self.window, self.max_energy]).all() or min(self.budget, self.window, self.max_energy) <= 0:
            raise ControllerError("invalid_calibration", "Invalid scalar calibration")
        fractions = np.array([a["fractions"] for a in self.actions], dtype=float)
        if fractions.shape != (6, 3) or not np.isfinite(fractions).all() or np.any(fractions < 0) or not np.allclose(fractions.sum(1), 1, atol=1e-8, rtol=0):
            raise ControllerError("invalid_actions", "Expected six normalized routing profiles")

    def validate_state(self, state):
        if not isinstance(state,DecisionState):
            raise ControllerError("invalid_state", "Expected a DecisionState record")
        try:
            q = np.array(state.queues, dtype=float)
            context = np.asarray(state.causal_values,dtype=float)
        except (TypeError,ValueError) as error:
            raise ControllerError("invalid_state", "Non-numeric state values") from error
        if state.schema_version != SCHEMA_VERSION or state.calibration_id != self.calibration_id:
            raise ControllerError("incompatible_state", "State schema/calibration mismatch")
        if q.shape != (3,) or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > self.buffer):
            raise ControllerError("invalid_state", "Queues must be within buffers")
        if not isinstance(state.demand_packets,Real) or not np.isfinite(state.demand_packets) or state.demand_packets < 0:
            raise ControllerError("invalid_state", "Demand must be finite and nonnegative")
        if state.split not in {"train", "validation", "test"} or not state.graph_file or not state.window_start:
            raise ControllerError("invalid_state", "Missing state identity")
        if not np.isfinite(context).all():
            raise ControllerError("invalid_state", "Non-finite causal context")

    def intervene(self, state, action):
        self.validate_state(state)
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 6:
            raise ControllerError("invalid_action", "Action must be an integer in 0..5")
        fractions = np.asarray(self.actions[action]["fractions"], dtype=float)
        q = np.asarray(state.queues, dtype=float)
        old = np.minimum(q, self.capacity)
        residual = q - old
        available = self.capacity - old
        arrivals = state.demand_packets * fractions
        current = np.minimum(arrivals, available)
        unserved = arrivals - current
        enqueued = np.minimum(unserved, np.maximum(self.buffer - residual, 0))
        drops = np.maximum(unserved - enqueued, 0)
        post = np.minimum(residual + enqueued, self.buffer)
        throughput = float((old + current).sum())
        pdr = float(np.clip(current.sum() / state.demand_packets, 0, 1)) if state.demand_packets else 1.0
        drop_ratio = float(np.clip(drops.sum() / state.demand_packets, 0, 1)) if state.demand_packets else 0.0
        weights = arrivals / state.demand_packets if state.demand_packets else fractions
        waiting = q / self.capacity * self.window * 1000
        latency = float(weights @ (self.base + waiting) + (1-pdr)*2*self.budget)
        utilization = (old + current) / self.capacity
        energy = float(np.sum(((q+arrivals)>0)*(self.idle+self.dynamic*utilization**2)))
        completion = float(pdr >= self.calibration["pdr_completion_threshold"] and latency <= self.budget)
        components = {"throughput": np.clip(throughput/self.capacity.sum(), 0, 1), "pdr": pdr,
                      "completion": completion, "latency": np.clip(latency/self.budget, 0, 1),
                      "energy": np.clip(energy/self.max_energy, 0, 1), "drops": drop_ratio}
        reward = float(np.clip(sum(self.environment["reward_weights"][k]*v for k,v in components.items()), -1, 1))
        intermediate = tuple((k, tuple(v.tolist())) for k,v in {
            "pre_queue":q, "arrivals":arrivals, "served_old":old, "residual_queue":residual,
            "remaining_capacity":available, "served_current":current, "unserved_current":unserved,
            "enqueued":enqueued, "drops":drops, "waiting_ms":waiting, "utilization":utilization}.items())
        return CounterfactualOutcome(int(action), self.actions[action]["name"], tuple(fractions), tuple(post),
                                     throughput, pdr, drop_ratio, latency, energy, float(drops.sum()), completion, reward, intermediate)

    def evaluate_actions(self, state):
        return tuple(self.intervene(state, a) for a in range(6))

    def specification(self):
        return {"schema_version":SCHEMA_VERSION, "mechanism_version":"phase4_equations_v1",
                "calibration":self.calibration, "environment":self.environment,
                "calibration_id":self.calibration_id, "units":{"latency_proxy_ms":"ms", "energy_proxy":"synthetic units"}}


def load_phase5_scm(artifact_dir):
    spec = json.loads((Path(artifact_dir)/"exact_scm.json").read_text())
    if spec["schema_version"] != SCHEMA_VERSION or spec["mechanism_version"] != "phase4_equations_v1":
        raise ControllerError("incompatible_model", "Unsupported SCM version")
    model = ActionAwareSCM(spec["calibration"], spec["environment"])
    if model.calibration_id != spec["calibration_id"]:
        raise ControllerError("incompatible_model", "Calibration digest mismatch")
    return model


class CounterfactualController:
    def __init__(self, scm, safety):
        self.scm = scm
        self.safety = dict(safety)
        self.limits = {"latency_proxy_ms":float(safety["latency_proxy_ms"]),
                       "energy_proxy":float(safety["energy_fraction"])*scm.max_energy,
                       "drop_ratio":float(safety["drop_ratio"]), "pdr":float(safety["pdr"])}
        self.tolerance = float(safety["tolerance"])
        if not np.isfinite(list(self.limits.values())+[self.tolerance]).all() or min(self.limits.values())<=0 or self.tolerance<0 or not 0 < safety["energy_fraction"] <= 1 or self.limits["pdr"]>1 or self.limits["drop_ratio"]>1:
            raise ControllerError("invalid_threshold", "Safety limits must be positive and valid")

    def assess(self, o):
        values = np.array([getattr(o,k) for k in METRICS]+list(o.post_queue), dtype=float)
        if not np.isfinite(values).all() or min(o.pdr,o.drop_ratio,o.latency_proxy_ms,o.energy_proxy,o.throughput_packets,o.dropped_packets)<0 or max(o.pdr,o.drop_ratio)>1 or o.energy_proxy>self.scm.max_energy+1e-8 or o.throughput_packets>self.scm.capacity.sum()+1e-8 or not -1<=o.reward<=1 or o.task_completion not in (0,1) or np.any(np.array(o.post_queue)<0) or np.any(np.array(o.post_queue)>self.scm.buffer+1e-8):
            raise ControllerError("invalid_prediction", "Prediction outside physical domain")
        reasons = tuple(k for k, limit in self.limits.items() if
                        (getattr(o,k)<limit-self.tolerance if k=="pdr" else getattr(o,k)>limit+self.tolerance))
        score = sum(max(0, (self.limits[k]-getattr(o,k))/self.limits[k] if k=="pdr" else getattr(o,k)/self.limits[k]-1) for k in self.limits)
        return reasons, float(score)

    def select_action(self, observation, state, q_values):
        try:
            observation = np.asarray(observation,dtype=float)
            q_values = np.asarray(q_values, dtype=float)
        except (TypeError,ValueError) as error:
            raise ControllerError("invalid_observation", "Non-numeric observation or Q-values") from error
        if observation.shape!=(67,) or not np.isfinite(observation).all() or q_values.shape!=(6,) or not np.isfinite(q_values).all():
            raise ControllerError("invalid_observation", "Expected finite observation and six Q-values")
        ranking = tuple(sorted(range(6), key=lambda a:(-q_values[a],a)))
        start = time.perf_counter()
        outcomes = self.scm.evaluate_actions(state)
        elapsed = time.perf_counter()-start
        assessments = tuple(self.assess(o) for o in outcomes)
        safe = [a for a in ranking if not assessments[a][0]]
        selected = safe[0] if safe else min(range(6), key=lambda a:(assessments[a][1],-q_values[a],a))
        status = "no_safe_action" if not safe else "unchanged" if selected==ranking[0] else "corrected"
        return SafetyDecision(ranking[0], selected, ranking, tuple(q_values), outcomes,
                              tuple(x[0] for x in assessments), tuple(x[1] for x in assessments), bool(safe), status, elapsed)

    def step(self, env, agent):
        if env.episode_steps >= env.episode_horizon or not env.valid_transition[env.current_index]:
            raise ControllerError("episode_finished", "Reset before another transition")
        if calibration_identity(env.calibration.to_dict())!=self.scm.calibration_id or env.config["environment"]!=self.scm.environment:
            raise ControllerError("incompatible_environment", "Environment differs from SCM")
        observation = env._observation()
        if not env.observation_space.contains(observation):
            raise ControllerError("invalid_observation", "Observation outside environment space")
        row = env.current_row
        state = DecisionState(float(row["total_packets"]), tuple(env.queues), env.data.split,
                              str(row["graph_file"]), row["window_start"].isoformat(), self.scm.calibration_id)
        agent.policy.set_training_mode(False)
        with torch.no_grad():
            tensor, _ = agent.policy.obs_to_tensor(observation)
            q_values = agent.q_net(tensor).cpu().numpy()[0]
        decision = self.select_action(observation, state, q_values)
        action,_ = agent.predict(observation, deterministic=True)
        if int(np.asarray(action).item())!=decision.proposed_action:
            raise ControllerError("ranking_mismatch", "Online Q-ranking differs from deterministic DQN")
        result = env.step(decision.executed_action)
        info = result[4]
        prediction = decision.outcomes[decision.executed_action]
        for name in METRICS:
            if not np.isclose(getattr(prediction,name),info[name],atol=1e-6,rtol=1e-8):
                raise ControllerError("realization_mismatch", f"Predicted/realized {name} differs")
        if not np.allclose(prediction.post_queue,info["post_queue"],atol=1e-6,rtol=1e-8):
            raise ControllerError("realization_mismatch", "Post-action queues differ")
        return result, decision, state
