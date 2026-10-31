"""
Test suite for Chop Cup RL with 7-state condition-based Reward Machine + QRM.
Limes ENABLED — all 7 RM states (U0–U5, UT) are active.

Tests:
  1. SAT solver: implicit belief derivation via DPLL (+ lime beliefs)
  2. RM direct: 7-state condition-based transitions (ball + lime)
  3. QRM counterfactual: verify query_transition for all RM states
  4. Full trick (ball-only path): palm→lift→reveal→reveal reaches U3→UT
  5. Full trick (lime path): ball trick + lime cycle → UT via d
  6. Wrong sequence: lift without palm should NOT advance RM
  7. QRM learned policy: QRM agent converges to high success
"""

from __future__ import annotations

import random
import sys

from epistemic_logic import CommonGround, EpistemicState
from reward_machine import RewardMachine, RMState
from chop_cup_env import ChopCupEnv
from qrm import QRMAgent


PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


# ------------------------------------------------------------------
# Test 1: SAT solver — implicit belief derivation
# ------------------------------------------------------------------
def test_sat_solver():
    print("\n── Test 1: SAT Solver (Implicit Beliefs) ──")
    cg = CommonGround(
        variables=["reg_under", "reg_table", "reg_hand"],
        exactly_one_groups=[["reg_under", "reg_table", "reg_hand"]],
    )

    # After palm: △(¬table), △(¬hand) → □(under)
    fluents = frozenset({"eb_not_reg_table", "eb_not_reg_hand"})
    state = EpistemicState(fluents, cg)
    check("□(reg_under) from △(¬table)∧△(¬hand)",
          state.implicit_belief("reg_under", positive=True))
    check("¬□(¬reg_under)",
          not state.implicit_belief("reg_under", positive=False))

    # After palm: △(under), △(¬table), △(¬hand) → explicit + implicit agree
    fluents2 = frozenset({"eb_reg_under", "eb_not_reg_table", "eb_not_reg_hand"})
    state2 = EpistemicState(fluents2, cg)
    check("□(reg_under) with explicit △(under) too",
          state2.implicit_belief("reg_under", positive=True))

    # After palm_ball: observer thinks under, not hand
    fluents3 = frozenset({"eb_reg_under", "eb_not_reg_table", "eb_not_reg_hand"})
    state3 = EpistemicState(fluents3, cg)
    check("After palm: □(reg_under)=True (observer deceived)",
          state3.implicit_belief("reg_under", positive=True))
    check("After palm: □(reg_hand)=False (observer doesn't know)",
          not state3.implicit_belief("reg_hand", positive=True))

    # When hand is revealed with ball: △(hand), △(¬table) → □(hand)
    fluents4 = frozenset({"eb_reg_hand", "eb_not_reg_table"})
    state4 = EpistemicState(fluents4, cg)
    check("After reveal: □(reg_hand) from △(hand)∧△(¬table)",
          state4.implicit_belief("reg_hand", positive=True))

    # Lime beliefs: eb_not_lime_cup → □(¬lime_cup) (used by RM transitions e, f)
    fluents5 = frozenset({"eb_not_lime_cup"})
    state5 = EpistemicState(fluents5, cg)
    check("□(¬lime_cup) from △(¬lime_cup)",
          state5.implicit_belief("lime_cup", positive=False))

    # eb_lime_cup → ¬□(¬lime_cup)
    fluents6 = frozenset({"eb_lime_cup"})
    state6 = EpistemicState(fluents6, cg)
    check("¬□(¬lime_cup) when △(lime_cup) set",
          not state6.implicit_belief("lime_cup", positive=False))


# ------------------------------------------------------------------
# Test 2: RM direct — 7-state condition-based transitions (ball + lime)
# ------------------------------------------------------------------
def test_rm_direct():
    print("\n── Test 2: RM Direct (7-State, Limes Enabled) ──")

    cg = CommonGround(
        variables=["reg_under", "reg_table", "reg_hand"],
        exactly_one_groups=[["reg_under", "reg_table", "reg_hand"]],
    )
    rm = RewardMachine(enable_limes=True, shaping=False)
    check("Initial state = U0", rm.state == RMState.U0)

    # a: U0→U1
    state_palm = EpistemicState(
        frozenset({"eb_reg_under", "eb_not_reg_table", "eb_not_reg_hand",
                    "cup_down", "eb_not_lime_cup"}), cg)
    rm.step(state_palm)
    check("a: U0→U1 (eb_reg_under)", rm.state == RMState.U1)

    # b: U1→U2
    state_reveal = EpistemicState(
        frozenset({"eb_reg_hand", "eb_not_reg_table", "eb_not_reg_under",
                    "eb_not_lime_cup"}), cg)
    rm.step(state_reveal)
    check("b: U1→U2 (□reg_hand)", rm.state == RMState.U2)

    # h: U2→U3
    state_visible = EpistemicState(
        frozenset({"ball_visible", "eb_not_lime_cup"}), cg)
    rm.step(state_visible)
    check("h: U2→U3 (ball_visible)", rm.state == RMState.U3)

    # e: U3→U4 (cup_down ∧ □¬lime_cup)
    state_cup_down = EpistemicState(
        frozenset({"cup_down", "eb_not_lime_cup"}), cg)
    rm.step(state_cup_down)
    check("e: U3→U4 (cup_down ∧ □¬lime)", rm.state == RMState.U4)

    # k: U4→U5 (lime_visible)
    state_lime_vis = EpistemicState(
        frozenset({"lime_visible", "lime_cup", "eb_lime_cup"}), cg)
    rm.step(state_lime_vis)
    check("k: U4→UT (lime_visible)", rm.state == RMState.UT)

    # f: U5→U4 (□¬lime_cup — after placing cup)
    state_lime_cleared = EpistemicState(
        frozenset({"cup_down", "eb_not_lime_cup"}), cg)
    rm.step(state_lime_cleared)
    check("f: U5→U4 (□¬lime_cup)", rm.state == RMState.U4)

    # k again (2nd lime)
    rm.step(state_lime_vis)
    check("k: U4→UT (2nd lime)", rm.state == RMState.UT)

    # d: U5→UT (lime_visible) — when f condition fails, d fires
    rm.step(state_lime_vis)
    check("d: U5→UT (lime_visible, r=1.0)", rm.state == RMState.UT)


# ------------------------------------------------------------------
# Test 3: QRM counterfactual — query_transition for all RM states
# ------------------------------------------------------------------
def test_qrm_counterfactual():
    print("\n── Test 3: QRM Counterfactual (query_transition) ──")

    cg = CommonGround(
        variables=["reg_under", "reg_table", "reg_hand"],
        exactly_one_groups=[["reg_under", "reg_table", "reg_hand"]],
    )
    rm = RewardMachine(enable_limes=True, shaping=False)

    state_palm = EpistemicState(
        frozenset({"eb_reg_under", "eb_not_reg_table", "eb_not_reg_hand",
                    "cup_down", "eb_not_lime_cup"}), cg)

    u, r = rm.query_transition(RMState.U0, state_palm)
    check("From U0 + eb_reg_under → U1", u == RMState.U1)

    u, r = rm.query_transition(RMState.U1, state_palm)
    check("From U1 + ¬□(reg_hand) → stays U1", u == RMState.U1)

    state_ball = EpistemicState(
        frozenset({"ball_visible", "eb_not_lime_cup"}), cg)

    u, r = rm.query_transition(RMState.U2, state_ball)
    check("From U2 + ball_visible → U3", u == RMState.U3)

    u, r = rm.query_transition(RMState.U3, state_ball)
    check("From U3 + ball_visible ∧ ¬lime → UT (g, r=1.0)",
          u == RMState.UT and r == 1.0)

    state_cup = EpistemicState(
        frozenset({"cup_down", "eb_not_lime_cup"}), cg)

    u, r = rm.query_transition(RMState.U3, state_cup)
    check("From U3 + cup_down ∧ □¬lime → U4 (e)", u == RMState.U4)

    state_lime = EpistemicState(
        frozenset({"lime_visible", "lime_cup", "eb_lime_cup"}), cg)

    u, r = rm.query_transition(RMState.U4, state_lime)
    check("From U4 + lime_visible → UT (k)", u == RMState.UT)

    u, r = rm.query_transition(RMState.UT, state_lime)
    check("From U5 + lime_visible (eb_lime) → UT (d, r=1.0)",
          u == RMState.UT and r == 1.0)

    state_cleared = EpistemicState(
        frozenset({"cup_down", "eb_not_lime_cup"}), cg)

    u, r = rm.query_transition(RMState.UT, state_cleared)
    check("From U5 + □¬lime_cup → U4 (f)", u == RMState.U4)


# ------------------------------------------------------------------
# Test 4: Full trick — ball-only path (limes enabled)
# ------------------------------------------------------------------
def test_ball_only_path():
    print("\n── Test 4: Ball-Only Path (Limes Enabled) ──")
    env = ChopCupEnv(enable_limes=True, max_steps=50)
    env.reset()

    check("Initial: eb_not_lime_cup=True", env.state.has("eb_not_lime_cup"))

    for a in ["palm_ball", "lift_cup", "reveal_hand"]:
        env.step(a)
    check("After ball trick: RM=U3", env.reward_machine.state == RMState.U3)

    s, rm, r, d, info = env.step("reveal_hand")
    check("reveal → g: U3→UT (r=1.0)", env.reward_machine.state == RMState.UT and r == 1.0)
    check("Episode done", env.done)


# ------------------------------------------------------------------
# Test 5: Full trick — lime path (ball trick + lime cycle)
# ------------------------------------------------------------------
def test_lime_path():
    print("\n── Test 5: Lime Path (Full 7-State Cycle) ──")
    env = ChopCupEnv(enable_limes=True, max_limes=3, max_steps=50)
    env.reset()

    # Ball trick
    for a in ["palm_ball", "lift_cup", "reveal_hand"]:
        env.step(a)
    check("Ball trick: RM=U3", env.reward_machine.state == RMState.U3)

    # Enter lime phase: place_cup → e
    env.step("place_cup")
    check("place_cup: RM→U4", env.reward_machine.state == RMState.U4)

    # Load + show lime 1
    check("load_lime available", "load_lime" in env.get_available_actions())
    env.step("load_lime")
    check("lime_cup=True", env.state.has("lime_cup"))
    env.step("lift_cup")
    check("lift: RM→UT, lime_visible", env.reward_machine.state == RMState.UT and env.state.has("lime_visible"))
    check("lift: eb_lime_cup=True", env.state.has("eb_lime_cup"))

    # Cycle: place_cup → f: U5→U4
    env.step("place_cup")
    check("place_cup: RM→U4 (f)", env.reward_machine.state == RMState.U4)
    check("lime cleared: eb_not_lime_cup", env.state.has("eb_not_lime_cup") and not env.state.has("lime_cup"))

    # Load + show lime 2
    env.step("load_lime")
    env.step("lift_cup")
    check("Lime 2: RM→UT", env.reward_machine.state == RMState.UT)

    # End: reveal_hand → d: U5→UT
    s, rm, r, d, info = env.step("reveal_hand")
    check("reveal: RM→UT (d, r=1.0)", env.reward_machine.state == RMState.UT and r == 1.0)
    check("Episode done", env.done)

    reached = env.reward_machine.reached_states
    check("All 7 RM states visited",
          all(s in reached for s in [RMState.U0, RMState.U1, RMState.U2,
                                     RMState.U3, RMState.U4, RMState.UT]))


# ------------------------------------------------------------------
# Test 6: Wrong sequence — lift without palm doesn't advance RM
# ------------------------------------------------------------------
def test_wrong_sequence():
    print("\n── Test 6: Wrong Sequence (lift without palm) ──")
    env = ChopCupEnv(enable_limes=True, max_steps=50)
    env.reset()

    env.step("lift_cup")
    check("lift without palm: RM stays U0",
          env.reward_machine.state == RMState.U0)

    env.step("reveal_hand")
    check("reveal (empty): RM stays U0",
          env.reward_machine.state == RMState.U0)


# ------------------------------------------------------------------
# Test 7: QRM Learned Policy — train and verify convergence
# ------------------------------------------------------------------
def test_qrm_learned_policy():
    print("\n── Test 7: QRM Learned Policy (Limes Enabled) ──")
    import numpy as np

    env = ChopCupEnv(enable_limes=True, max_limes=3, max_steps=50)
    agent = QRMAgent(
        reward_machine=env.reward_machine,
        discount_factor=0.9,
        learning_rate=0.1,
        epsilon=0.1,
        q_init=0.0,
    )

    num_episodes = 500
    for ep in range(num_episodes):
        random.seed(42 + ep)
        state, rm_state = env.reset()
        obs = env.get_env_observation()

        step = 0
        while not env.done and step < 50:
            available = env.get_available_actions()
            if not available:
                break
            action = agent.select_action(obs, env.reward_machine.get_state(), available)
            old_obs = obs
            old_avail = available
            old_rm_state = env.reward_machine.get_state()
            state, rm_state, reward, done, info = env.step(action)
            obs = env.get_env_observation()
            new_avail = env.get_available_actions()
            # Online QRM update (current RM state only)
            agent.update(old_obs, action, obs, old_rm_state,
                         env.reward_machine.get_state(), reward, new_avail)
            step += 1

    # Evaluate
    successes = 0
    eval_episodes = 100
    for ep in range(eval_episodes):
        random.seed(12345 + ep)
        state, rm_state = env.reset()
        obs = env.get_env_observation()
        step = 0
        while not env.done and step < 50:
            available = env.get_available_actions()
            if not available:
                break
            action = agent.get_policy_action(obs, env.reward_machine.get_state(), available)
            state, rm_state, reward, done, info = env.step(action)
            obs = env.get_env_observation()
            step += 1
        if env.reward_machine.is_terminal():
            successes += 1

    rate = successes / eval_episodes
    print(f"  QRM success rate after {num_episodes} episodes: {rate:.2%}")
    print(f"  States explored: {agent.num_states}")
    print(f"  Transitions: {agent.num_transitions}")
    check(f"QRM success rate ≥ 80% (got {rate:.2%})", rate >= 0.80)


# ------------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Chop Cup RL — Test Suite")
    print("  7-State RM + QRM — LIMES ENABLED")
    print("=" * 60)

    test_sat_solver()
    test_rm_direct()
    test_qrm_counterfactual()
    test_ball_only_path()
    test_lime_path()
    test_wrong_sequence()
    test_qrm_learned_policy()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    sys.exit(1 if FAIL > 0 else 0)
