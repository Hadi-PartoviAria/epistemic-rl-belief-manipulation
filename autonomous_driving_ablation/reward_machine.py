"""Reward Machine -- Waymo-style urban driving with epistemic ambiguity.

Physical-only RM: all transition guards are observable physical/compliance
facts. Implicit beliefs (Box(drivable), Box(lane_closed), ...) are computed
inside BEQRM and added to its observation, but never appear in any RM guard.

States
------
  u0  Start       vehicle is at A, has not yet moved
  u1  Navigating  vehicle is driving toward B on clear road
  u2  Ambiguous   vehicle is at a cell with ambiguous visual label
  u3  Resolved    vehicle just handled an ambiguous cell correctly
  uT  Success     reached B safely and legally  (+1.0)
  uF  Failure     entered a truly non-drivable cell (0.0)

Event labels (single chars emitted by env.get_events())
-------------------------------------------------------
  a = left_A          vehicle moved out of the start cell
  b = at_ambiguous    current cell has ambiguous visual (water_like/cones/sign)
  c = resolved        this step the vehicle correctly handled an ambiguous cell
  d = continues       at a non-ambiguous cell, safe, legal
  e = success         reached_B AND safe AND legal
  f = unsafe_entry    just entered a truly non-drivable cell
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


class RMState(IntEnum):
    U0 = 0
    U1 = 1
    U2 = 2
    U3 = 3
    UT = -1
    UF = -2


def evaluate_dnf(formula: str, true_props: str) -> bool:
    if "|" in formula:
        return any(evaluate_dnf(f, true_props) for f in formula.split("|"))
    if "&" in formula:
        return all(evaluate_dnf(f, true_props) for f in formula.split("&"))
    if formula.startswith("!"):
        return not evaluate_dnf(formula[1:], true_props)
    if formula == "True":
        return True
    if formula == "False":
        return False
    return formula in true_props


class RewardFunction:
    def get_reward(self, s_info=None) -> float:
        raise NotImplementedError

    def get_type(self) -> str:
        raise NotImplementedError


class ConstantRewardFunction(RewardFunction):
    def __init__(self, c: float):
        self.c = c

    def get_type(self) -> str:
        return "constant"

    def get_reward(self, s_info=None) -> float:
        return self.c


def value_iteration(U, delta_u, delta_r, terminal_u, gamma):
    V = {u: 0.0 for u in U}
    V[terminal_u] = 0.0
    err = 1.0
    while err > 1e-7:
        err = 0.0
        for u1 in U:
            qs = [rf.get_reward() + gamma * V.get(u2, 0.0)
                  for u2, rf in delta_r[u1].items()]
            v_new = max(qs) if qs else 0.0
            err = max(err, abs(v_new - V[u1]))
            V[u1] = v_new
    return V


class RewardMachine:
    """Waymo driving RM -- physical-only guards."""

    def __init__(self):
        self.U: List[RMState] = [RMState.U0, RMState.U1, RMState.U2, RMState.U3]
        self.u0 = RMState.U0
        self.terminal_u = RMState.UT
        self.failure_u = RMState.UF
        self.delta_u: Dict[RMState, Dict[RMState, str]] = {}
        self.delta_r: Dict[RMState, Dict[RMState, RewardFunction]] = {}
        self.known_transitions: Dict[Tuple[RMState, str], RMState] = {}
        self._build_transitions()

        self.state = self.u0
        self.total_reward = 0.0
        self.transition_history: List[str] = []
        self.reached_states: Set[RMState] = {self.u0}

        self.gamma = 1.0
        self.potentials: Dict[RMState, float] = {u: 0.0 for u in self.U}
        self.potentials[self.terminal_u] = 0.0
        self.potentials[self.failure_u] = 0.0

    def _add(self, u1, u2, dnf, reward):
        self.delta_u.setdefault(u1, {})[u2] = dnf
        self.delta_r.setdefault(u1, {})[u2] = ConstantRewardFunction(reward)

    def _build_transitions(self):
        self._add(RMState.U0, RMState.U1, "a", 0.0)

        self._add(RMState.U1, RMState.UT, "e", 1.0)   # success (highest priority)
        self._add(RMState.U1, RMState.UF, "f", 0.0)   # failure
        self._add(RMState.U1, RMState.U2, "b", 0.0)   # enter ambiguous

        self._add(RMState.U2, RMState.UF, "f", 0.0)   # unsafe from ambiguous
        self._add(RMState.U2, RMState.U3, "c", 0.0)   # resolved correctly

        self._add(RMState.U3, RMState.UF, "f", 0.0)
        self._add(RMState.U3, RMState.U1, "d", 0.0)   # continue navigating

    def add_reward_shaping(self, gamma, rs_gamma):
        self.gamma = gamma
        V = value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_u, rs_gamma)
        self.potentials = {u: -V[u] for u in V}
        self.potentials[self.terminal_u] = 0.0
        self.potentials[self.failure_u] = 0.0

    def _compute_next_state(self, u1, true_props):
        for u2 in self.delta_u.get(u1, {}):
            if evaluate_dnf(self.delta_u[u1][u2], true_props):
                return u2
        return u1

    def get_next_state(self, u1, true_props):
        key = (u1, true_props)
        if key not in self.known_transitions:
            self.known_transitions[key] = self._compute_next_state(u1, true_props)
        return self.known_transitions[key]

    def query_transition(self, u1, true_props):
        if u1 in (self.terminal_u, self.failure_u):
            return u1, 0.0
        u2 = self.get_next_state(u1, true_props)
        if u2 == u1:
            return u1, 0.0
        return u2, self.delta_r[u1][u2].get_reward()

    def step(self, true_props):
        total_r = 0.0
        info_parts = []
        terminal = False
        fired: Set[Tuple[RMState, RMState]] = set()
        for _ in range(20):
            u1 = self.state
            if u1 in (self.terminal_u, self.failure_u):
                terminal = True
                break
            u2 = None
            for cand in self.delta_u.get(u1, {}):
                if (u1, cand) in fired:
                    continue
                if evaluate_dnf(self.delta_u[u1][cand], true_props):
                    u2 = cand
                    break
            if u2 is None or u2 == u1:
                break
            fired.add((u1, u2))
            r = self.delta_r[u1][u2].get_reward()
            self.state = u2
            self.total_reward += r
            total_r += r
            self.transition_history.append(f"{u1.name.lower()}->{u2.name.lower()}")
            self.reached_states.add(u2)
            info_parts.append(f"{u1.name.lower()}=>{u2.name.lower()}")
            if u2 in (self.terminal_u, self.failure_u):
                terminal = True
                break
        info = ("RM: " + " ".join(info_parts) + f" r={total_r}") if info_parts else ""
        return total_r, terminal, info

    def reset(self):
        self.state = self.u0
        self.total_reward = 0.0
        self.transition_history = []
        self.reached_states = {self.u0}

    def get_state(self):
        return self.state

    def get_state_index(self):
        return {RMState.U0: 0, RMState.U1: 1, RMState.U2: 2,
                RMState.U3: 3, RMState.UT: 4, RMState.UF: 5}.get(self.state, 0)

    def get_states(self):
        return list(self.U)

    def get_all_states(self):
        return list(self.U) + [self.terminal_u, self.failure_u]

    def is_terminal(self):
        return self.state in (self.terminal_u, self.failure_u)
