#!/usr/bin/env python3
"""
=============================================================================
  INTERACTIVE TUTORIAL — Understanding the Epistemic RL Chop Cup Codebase
=============================================================================

Run this file:
    cd chop_cup_rl
    python tutorial.py

It walks you through every layer of the code, printing outputs so you can see
exactly what each class and function does.  Read the comments, then look at the
printed output.

The layers (bottom → top):
  1. epistemic_logic.py  — SAT solver + belief state
  2. reward_machine.py   — finite automaton tracking trick phases
  3. chop_cup_env.py     — the environment (actions + effects)
  4. qrm.py              — the Q-learning agent
  5. main.py             — the training loop
=============================================================================
"""

import sys, random
import numpy as np

# ── helpers ──
def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

import os
INTERACTIVE = sys.stdin.isatty() and "TUTORIAL_NOPAUSE" not in os.environ

def pause():
    if INTERACTIVE:
        input("\n  ▸ Press Enter to continue...\n")
    else:
        print()  # non-interactive: just print a blank line

# ===================================================================
#  LAYER 1 — epistemic_logic.py
# ===================================================================
banner("LAYER 1: epistemic_logic.py  —  The SAT Solver & Belief State")
print("""
This file has three pieces:
  (a) A tiny SAT solver  (DPLL algorithm)
  (b) CommonGround       — background knowledge the Observer has
  (c) EpistemicState      — the full state: physical facts + explicit beliefs

Let's explore each one.
""")

# ── 1a. SAT solver ──────────────────────────────────────────────────
print("─" * 50)
print("1a. The SAT Solver")
print("─" * 50)
print("""
A SAT solver answers: "Given a set of logical clauses, is there ANY
assignment of True/False to the variables that makes them ALL true?"

The code uses CNF (Conjunctive Normal Form):
  - Each variable can be True or False
  - A 'literal' is (variable_name, is_positive), e.g. ("x", True) means x
  - A 'clause' is a set of literals joined by OR
  - The whole formula is all clauses joined by AND

Example:   (x OR ¬y)  AND  (y OR z)  AND  (¬x OR ¬z)
""")

from epistemic_logic import make_literal, is_satisfiable

# Let's test: (x) AND (¬x)  — clearly UNSAT
clause1 = {make_literal("x", True)}    # {("x", True)}   means: x must be true
clause2 = {make_literal("x", False)}   # {("x", False)}  means: x must be false
result = is_satisfiable([clause1, clause2])
print(f"  (x) AND (¬x)  → satisfiable? {result}")   # False — contradiction!

# (x OR y) AND (¬x OR y) — SAT: y=True works
clause1 = {make_literal("x", True), make_literal("y", True)}
clause2 = {make_literal("x", False), make_literal("y", True)}
result = is_satisfiable([clause1, clause2])
print(f"  (x∨y) AND (¬x∨y)  → satisfiable? {result}")  # True

pause()

# ── 1b. CommonGround ────────────────────────────────────────────────
print("─" * 50)
print("1b. CommonGround — The Observer's Background Knowledge")
print("─" * 50)
print("""
The Observer believes: "the ball is in EXACTLY ONE of three places:
  reg_under (under cup), reg_table (on table), reg_hand (in hand)"

This is an XOR constraint, encoded as CNF:
  - At-least-one:  (reg_under ∨ reg_table ∨ reg_hand)
  - At-most-one:   (¬reg_under ∨ ¬reg_table)
                    (¬reg_under ∨ ¬reg_hand)
                    (¬reg_table ∨ ¬reg_hand)
""")

from epistemic_logic import CommonGround

cg = CommonGround(
    variables=["reg_under", "reg_table", "reg_hand"],
    exactly_one_groups=[["reg_under", "reg_table", "reg_hand"]],
)

print("  CommonGround clauses:")
for i, clause in enumerate(cg.get_clauses()):
    readable = []
    for var, pos in clause:
        readable.append(var if pos else f"¬{var}")
    print(f"    Clause {i}: ({' ∨ '.join(readable)})")

pause()

# ── 1c. EpistemicState ──────────────────────────────────────────────
print("─" * 50)
print("1c. EpistemicState — Physical Facts + Observer's Beliefs")
print("─" * 50)
print("""
The state is a frozenset of strings. Each string is either:
  - A physical fact:     "reg_table", "cup_down", "mag_stuck"
  - An explicit belief:  "eb_reg_table"   means Δ(reg_table) — Observer saw ball on table
                         "eb_not_reg_hand" means Δ(¬reg_hand) — Observer saw hand is empty

Implicit beliefs (□φ) are NOT stored. They are computed on-demand:
  □φ  iff  CommonGround ∧ ExplicitBeliefs ∧ ¬φ  is UNSAT
""")

from epistemic_logic import EpistemicState

# Initial state: ball is on the table, cup is down, magnetic ball stuck
initial = EpistemicState(
    fluents=frozenset({"mag_stuck", "reg_table", "cup_down", "eb_reg_table"}),
    common_ground=cg,
)

print("  Initial state fluents:", sorted(initial.fluents))
print()
print("  Checking physical facts:")
print(f"    has('reg_table')  = {initial.has('reg_table')}")   # True
print(f"    has('reg_under')  = {initial.has('reg_under')}")   # False
print(f"    has('reg_hand')   = {initial.has('reg_hand')}")    # False
print()
print("  Checking explicit beliefs (Δ):")
print(f"    has('eb_reg_table')  = {initial.has('eb_reg_table')}")  # True — Observer sees ball on table
print(f"    has('eb_reg_under')  = {initial.has('eb_reg_under')}")  # False
print()
print("  Checking implicit beliefs (□) via SAT:")
print(f"    □(reg_table)?  = {initial.implicit_belief('reg_table', positive=True)}")
print(f"    □(reg_under)?  = {initial.implicit_belief('reg_under', positive=True)}")
print(f"    □(reg_hand)?   = {initial.implicit_belief('reg_hand', positive=True)}")

print("""
  ↑ The SAT solver checked: CG ∧ Δ(reg_table) ∧ ¬reg_table → is it UNSAT?
    Yes → so □(reg_table) = True.
    The Observer can logically DEDUCE the ball is on the table.
""")

pause()

# ── 1c'. The deception moment ──
print("─" * 50)
print("1c'. THE KEY MOMENT: How False Belief Arises")
print("─" * 50)
print("""
After the magician palms the ball, the beliefs change but NOT reality:
  Physical:  reg_hand = True  (ball is secretly in hand)
  Beliefs:   Δ(reg_under) = True     ← transferred from Δ(reg_table)
             Δ(¬reg_table) = True
             Δ(¬reg_hand)  = True    ← Observer didn't see the palm
""")

deceived = EpistemicState(
    fluents=frozenset({
        "mag_stuck", "reg_hand", "cup_down",    # Physical: ball in HAND
        "eb_reg_under",                          # Observer thinks: under cup
        "eb_not_reg_table",                      # Observer thinks: not on table
        "eb_not_reg_hand",                       # Observer thinks: not in hand
    }),
    common_ground=cg,
)

print("  Physical reality:")
print(f"    reg_hand  = {deceived.has('reg_hand')}")    # True — ball IS in hand
print(f"    reg_under = {deceived.has('reg_under')}")   # False — nothing under cup
print()
print("  Observer's explicit beliefs:")
print(f"    Δ(reg_under)   = {deceived.has('eb_reg_under')}")     # True
print(f"    Δ(¬reg_table)  = {deceived.has('eb_not_reg_table')}")  # True
print(f"    Δ(¬reg_hand)   = {deceived.has('eb_not_reg_hand')}")   # True
print()
print("  SAT-derived implicit beliefs:")
ib_under = deceived.implicit_belief("reg_under", positive=True)
ib_hand = deceived.implicit_belief("reg_hand", positive=True)
print(f"    □(reg_under) = {ib_under}")   # True — Observer DEDUCES ball is under cup
print(f"    □(reg_hand)  = {ib_hand}")    # False — Observer has no reason to think hand
print()

if ib_under and not deceived.has("reg_under"):
    print("  ★ FALSE BELIEF ACHIEVED! ★")
    print("    Observer believes □(reg_under) = True")
    print("    But physical reg_under = False")
    print("    The deception is: □φ ∧ ¬φ")

pause()

# ===================================================================
#  LAYER 2 — reward_machine.py
# ===================================================================
banner("LAYER 2: reward_machine.py  —  The Finite Automaton")
print("""
The Reward Machine (RM) tracks which PHASE of the trick the agent is in.
It's a state machine:

   u0 (Setup)  →  u1 (Covered)  →  u2 (Pocket)  →  u3 (Appear)
                                                       ↓     ↓
                                                      u0    uT (+1)
                                                            ↕
                                                           u4 (+10 for lime)

Each transition has a CONDITION (a function that checks the epistemic state).
The RM only advances when the condition is met.
""")

from reward_machine import RewardMachine, RMState

rm = RewardMachine(enable_limes=True, shaping=False)

print("  All RM states:", [s.value for s in rm.get_all_states()])
print(f"  Current state: {rm.get_state().value}")
print()
print("  Transitions:")
for t in rm.transitions:
    print(f"    {t.source.value} → {t.target.value}  "
          f"[label={t.label}, reward={t.reward:+.1f}]  {t.description}")

pause()

print("─" * 50)
print("2a. Stepping the RM — Condition Evaluation")
print("─" * 50)
print("""
The RM's step() method checks each transition's condition against the
current epistemic state. If a condition is satisfied, the RM advances.
""")

rm.reset()
print(f"  RM state: {rm.get_state().value}")

# Simulate: after place_cup with observer believing ball is under cup
state_u1 = EpistemicState(
    fluents=frozenset({"mag_stuck", "cup_down", "eb_reg_under", "eb_not_reg_table", "eb_not_reg_hand"}),
    common_ground=cg,
)
reward, terminal, desc = rm.step(state_u1)
print(f"  → Stepped with Δ(reg_under)=True")
print(f"    RM transition: {desc}")
print(f"    Reward: {reward}, New state: {rm.get_state().value}")

pause()

# ===================================================================
#  LAYER 3 — chop_cup_env.py
# ===================================================================
banner("LAYER 3: chop_cup_env.py  —  The Environment")
print("""
ChopCupEnv ties everything together:
  - Manages the physical+belief state (EpistemicState)
  - Applies action effects (lift_cup, place_cup, palm_ball, reveal_hand)
  - Calls reward_machine.step() after each action
  - Provides observations for the RL agent

Let's watch a full episode unfold step by step.
""")

from chop_cup_env import ChopCupEnv

env = ChopCupEnv(enable_limes=True, max_limes=3, max_steps=50)
state, rm_state = env.reset()

print(f"  After reset:")
print(f"    State: {env.get_compact_state_str()}")
print(f"    RM state: {rm_state.value}")
print(f"    Available actions: {env.get_available_actions()}")

pause()

print("─" * 50)
print("3a. Executing the Trick Step by Step")
print("─" * 50)

# The optimal trick sequence:
trick_sequence = [
    ("palm_ball",   "SECRET: ball goes table→hand, Observer thinks table→under"),
    ("place_cup",   "Put cup down; belief transfers further"),
    ("lift_cup",    "Lift cup — magnetic ball appears! Observer is shocked"),
    ("place_cup",   "Put cup back down (reset cycle, RM: u3→u0 if needed)"),
]

for action, explanation in trick_sequence:
    avail = env.get_available_actions()
    if action not in avail:
        print(f"\n  ✗ {action} not available (available: {avail})")
        print(f"    Skipping... (this is normal — guards prevent invalid actions)")
        continue

    print(f"\n  ▸ Action: {action}")
    print(f"    Why: {explanation}")
    state, rm_state, reward, done, info = env.step(action)
    print(f"    Result: {info}")
    print(f"    State:  {env.get_compact_state_str()}")
    print(f"    RM:     {rm_state.value} | Reward: {reward:+.1f}")
    print(f"    Available next: {env.get_available_actions()}")

pause()

# ===================================================================
#  LAYER 4 — qrm.py
# ===================================================================
banner("LAYER 4: qrm.py  —  The Q-Learning Agent (QRM)")
print("""
QRM = Q-Learning for Reward Machines (Icarte et al., ICML 2018).

Key idea: maintain separate Q-values for each RM state.
  Q-table is indexed by: (env_observation, rm_state, action)

The agent:
  1. Observes (env_state, rm_state)
  2. Picks action via ε-greedy
  3. Gets reward from the RM
  4. Updates Q-value for (obs, rm_state, action)

Let's create an agent and watch it learn from a few episodes.
""")

from qrm import QRMAgent

env2 = ChopCupEnv(enable_limes=True, max_limes=3, max_steps=50)
agent = QRMAgent(
    reward_machine=env2.reward_machine,
    discount_factor=0.9,
    learning_rate=0.1,
    epsilon=0.3,     # high exploration for demo
    q_init=0.0,
)

print(f"  Agent created:")
print(f"    γ (discount):  {agent.gamma}")
print(f"    α (lr):        {agent.lr}")
print(f"    ε (explore):   {agent.epsilon}")
print(f"    Q-table size:  {agent.num_states} states (empty — hasn't seen anything)")

pause()

print("─" * 50)
print("4a. Training for 200 Episodes — Watch Q-table Grow")
print("─" * 50)

random.seed(42)
np.random.seed(42)

for ep in range(1, 201):
    state, rm_state = env2.reset()
    obs = env2.get_env_observation()
    step = 0

    while not env2.done and step < 50:
        avail = env2.get_available_actions()
        if not avail:
            break

        action = agent.select_action(obs, env2.reward_machine.get_state(), avail)
        old_obs = obs
        old_rm = env2.reward_machine.get_state()

        state, rm_state, reward, done, info = env2.step(action)
        obs = env2.get_env_observation()
        new_avail = env2.get_available_actions()

        # This is the Q-learning update:
        #   Q[(obs, rm), a] += α * (r + γ * max Q[(obs', rm'), a'] - Q[(obs, rm), a])
        agent.update(old_obs, action, obs, old_rm, env2.reward_machine.get_state(),
                     reward, new_avail)
        step += 1

    if ep in [1, 5, 10, 50, 100, 200]:
        print(f"  Episode {ep:>3d}: Q-table has {agent.num_states} states, "
              f"{agent.num_transitions} total updates")

pause()

print("─" * 50)
print("4b. Q-Table Sample — What the Agent Learned")
print("─" * 50)
print()

# Show a few interesting Q-values
count = 0
for state_key in sorted(agent.Q.keys(), key=str):
    actions = agent.Q[state_key]
    # Only show states where at least one Q-value differs from 0
    interesting = {a: q for a, q in actions.items() if abs(q) > 0.01}
    if interesting and count < 8:
        print(f"  State: {state_key}")
        for a, q in sorted(interesting.items(), key=lambda x: -x[1]):
            marker = " ◄ best" if q == max(interesting.values()) else ""
            print(f"    Q({a:>12s}) = {q:+.4f}{marker}")
        print()
        count += 1

pause()

# ===================================================================
#  LAYER 5 — main.py
# ===================================================================
banner("LAYER 5: main.py  —  The Training Loop")
print("""
main.py orchestrates everything:
  1. Creates the environment (ChopCupEnv)
  2. Creates the agent (QRMAgent)
  3. Runs a training loop:
       For each episode:
         Reset env → get (state, rm_state)
         While not done:
           Pick action (ε-greedy)
           Step env → get (new_state, new_rm_state, reward)
           Update Q-table
         Periodically evaluate (greedy policy, no exploration)
  4. Saves results to CSV + generates plots

To run the full training:
    python main.py --episodes 500 --enable-limes

To run a quick test:
    python main.py --episodes 50 --no-limes --seed 42
""")

pause()

# ===================================================================
#  LAYER 6 — Running a mini-training & seeing the result
# ===================================================================
banner("LAYER 6: Mini-Training — Watch it All Work Together")

from main import train, evaluate

print("  Training for 100 episodes (this should take a few seconds)...\n")
agent_trained, stats = train(
    num_episodes=100,
    eval_interval=10,
    eval_episodes=50,
    max_steps=50,
    discount_factor=0.9,
    learning_rate=0.1,
    epsilon=0.1,
    enable_limes=True,
    max_limes=3,
    seed=42,
    output_dir="/tmp/tutorial_results",
    verbose=True,
)

pause()

# ===================================================================
#  LAYER 7 — Watching the Learned Policy Execute
# ===================================================================
banner("LAYER 7: Watching the Learned Policy")
print("  Let's watch the trained agent perform the trick (greedy, no exploration):\n")

env_demo = ChopCupEnv(enable_limes=True, max_limes=3, max_steps=20)
state, rm_state = env_demo.reset()
obs = env_demo.get_env_observation()

print(f"  Step 0: {env_demo.get_compact_state_str()}")

for step in range(1, 21):
    avail = env_demo.get_available_actions()
    if not avail or env_demo.done:
        break

    action = agent_trained.get_policy_action(obs, env_demo.reward_machine.get_state(), avail)
    state, rm_state, reward, done, info = env_demo.step(action)
    obs = env_demo.get_env_observation()

    reward_str = f"  ★ reward={reward:+.1f}" if reward != 0 else ""
    print(f"  Step {step:>2d}: {action:<14s} → RM={rm_state.value}  "
          f"{env_demo.get_compact_state_str()}{reward_str}")

print(f"\n  Final RM states reached: {[s.value for s in env_demo.reward_machine.reached_states]}")
print(f"  Total reward: {env_demo.reward_machine.total_reward:.1f}")

# ===================================================================
banner("DONE — Code Architecture Summary")
print("""
  File dependency graph (bottom to top):

    epistemic_logic.py    ← SAT solver, CommonGround, EpistemicState
          ↑
    reward_machine.py     ← RM states, transitions, conditions
          ↑
    chop_cup_env.py       ← Actions, guards, effects, observation
          ↑
    qrm.py                ← Q-table, ε-greedy, Q-update
          ↑
    main.py               ← Training loop, evaluation, plots, CLI

  The flow for ONE step:
    1. Agent calls  agent.select_action(obs, rm_state, available)
    2. Environment  env.step(action)
       a. Apply physical + belief effects  → new EpistemicState
       b. RM checks conditions on new state → maybe transitions
       c. Return (state, rm_state, reward, done, info)
    3. Agent calls  agent.update(obs, action, obs', rm, rm', reward, ...)
       → Q[(obs,rm)][action] += α * (r + γ * max Q[(obs',rm')][a'] - old_Q)

  Next steps to explore:
    • python main.py --episodes 500              Full training with limes
    • python main.py --no-limes --episodes 200   Ball-only mode
    • python run_comparison.py                    Compare all methods
    • python test_epistemic.py                    Unit tests for SAT solver
""")
