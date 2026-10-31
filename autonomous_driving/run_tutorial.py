"""Quick tutorial / smoke test for the Autonomous Driving environment (5x5).

Demonstrates:
  1. Building the environment and Common Ground
  2. Following a rule-compliant route from A=(1,5) to B=(5,1)
  3. Verifying SAT-based implicit beliefs at each step
  4. Reaching the success state uT
"""

from __future__ import annotations

from autonomous_driving_env import AutonomousDrivingEnv, START, GOAL
from reward_machine import RMState


def run_compliant_route():
    """Run a rule-compliant route: east to column 5, then south to row 1.

    Route:
      (1,5) → east → (5,5): 4 east moves on priority road (no stops)
      Turn south
      (5,5) → south → (5,1): 4 south moves on ordinary road
        Each ordinary cell: stop once, then move.
    """
    print("=" * 72)
    print("  AUTONOMOUS DRIVING TUTORIAL (5x5)")
    print("=" * 72)

    env = AutonomousDrivingEnv(max_steps=100, initial_heading="east")
    env.reset()

    print(f"\nInitial state: {env.get_compact_state_str()}")
    print(f"  RM state: {env.reward_machine.get_state().value}")

    def step_and_report(env, action: str) -> None:
        _, rm_state, reward, done, info = env.step(action)
        print(f"  step {env.step_count:3d} | action={action:14s} | "
              f"{env.get_compact_state_str()} | r={reward:.2f}"
              + (f" | {info}" if info else ""))

    print("\n--- Phase 1: East on priority road (no stops needed) ---")
    # Move east 4 times — visiting cells (1,5), (2,5), (3,5), (4,5)
    for _ in range(4):
        step_and_report(env, "move_forward")
        if env.done:
            break

    print("\n--- Phase 2: Turn south ---")
    step_and_report(env, "turn_right")  # east → south

    print("\n--- Phase 3: South on ordinary road (stop once per cell) ---")
    # From (5,5), move south to (5,1) stopping once per cell
    for _ in range(4):
        if env.done:
            break
        step_and_report(env, "stop")
        if env.done:
            break
        step_and_report(env, "move_forward")

    print("\n" + "=" * 72)
    print("  FINAL RESULT")
    print("=" * 72)
    print(f"  Final position:    {env.pos}")
    print(f"  Reached goal:      {env.pos == GOAL}")
    print(f"  Total reward:      {env.reward_machine.total_reward:.2f}")
    print(f"  Final RM state:    {env.reward_machine.get_state().value}")
    print(f"  Reached uT:        {RMState.UT in env.reward_machine.reached_states}")
    print(f"  RM transitions:    {' '.join(env.reward_machine.transition_history)}")
    print(f"  Visited cells:     {len(env.visited_cells)}")
    print(f"  Physical rule_compliant: {env.state.has('rule_compliant')}")
    print(f"  Observer □rule_compliant: "
          f"{env.state.has('observer_route_compliant')}")
    print("=" * 72)


def run_violation_route():
    """Run a route with a rule violation (no stop on ordinary road)."""
    print("\n\n" + "=" * 72)
    print("  RULE VIOLATION SCENARIO")
    print("=" * 72)
    print("  Going east then south without stopping (ordinary-road violation)")

    env = AutonomousDrivingEnv(max_steps=200, initial_heading="east")
    env.reset()

    # Go east 2 cells then turn south and run through without stopping
    for _ in range(2):
        env.step("move_forward")
    env.step("turn_right")  # face south
    # Now move south WITHOUT stopping — violation on ordinary road
    for _ in range(3):
        env.step("move_forward")
        if env.done:
            break

    print(f"  Final position:    {env.pos}")
    print(f"  rule_compliant:    {env.state.has('rule_compliant')}")
    print(f"  rule_violated:     {env.state.has('rule_violated')}")
    print(f"  Final RM state:    {env.reward_machine.get_state().value}")
    print(f"  Total reward:      {env.reward_machine.total_reward:.2f}")
    print("=" * 72)


if __name__ == "__main__":
    run_compliant_route()
    run_violation_route()
