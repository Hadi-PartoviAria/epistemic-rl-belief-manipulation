"""Autonomous Driving Environment -- Waymo-style urban driving with epistemic ambiguity.

7x7 grid from A=(1,7) to B=(7,1). Some cells have visually ambiguous labels
(water-like surfaces, orange cones) that traffic cars disambiguate via their
observed behaviour. BEQRM uses SAT-derived implicit beliefs (b_impl) to infer
whether to enter ambiguous cells; baseline methods (HRL-RM, DDQN) must learn
this mapping from sparse terminal reward alone.

Cell layout (GRID[y-1][x-1]):
    (1,7) ROAD   (2,7) PUDDLE  (3,7) ROAD   (4,7) ROAD   (5,7) ROAD   (6,7) ROAD   (7,7) ROAD
    (1,6) ROAD   (2,6) ROAD    (3,6) ROAD   (4,6) FLOODED(5,6) ROAD   (6,6) ROAD   (7,6) ROAD
    (1,5) ROAD   (2,5) ROAD    (3,5) ROAD   (4,5) ROAD   (5,5) CONSTR.(6,5) ROAD   (7,5) ROAD
    (1,4) ROAD   (2,4) ROAD    (3,4) PUDDLE (4,4) ROAD   (5,4) ROAD   (6,4) ROAD   (7,4) ROAD
    (1,3) ROAD   (2,3) ROAD    (3,3) ROAD   (4,3) ROAD   (5,3) ROAD   (6,3) CLOSED (7,3) ROAD
    (1,2) ROAD   (2,2) FLOODED (3,2) ROAD   (4,2) ROAD   (5,2) ROAD   (6,2) ROAD   (7,2) ROAD
    (1,1) ROAD   (2,1) ROAD    (3,1) ROAD   (4,1) CONSTR.(5,1) ROAD   (6,1) ROAD   (7,1) GOAL

Ambiguous visuals:
    water_like  -- could be shallow puddle (safe) or flooded (unsafe)
    cones       -- could be construction zone (slow but safe) or lane closed

Traffic simulation:
    N_TRAFFIC_CARS cars know true drivability and move each step.
    They generate evidence: cars_pass, cars_avoid, cars_slow per cell.
    After WARMUP_STEPS, ambiguous cells have enough evidence for SAT to
    derive definitive Box(drivable) or Box(not_drivable) verdicts.

RM event labels (physical-only, emitted by get_events()):
    a = left_A            AV departed start cell
    b = at_ambiguous      current cell has water_like/cones visual (drivable)
    c = resolved          just left an ambiguous cell safely
    d = continues         at a safe clear cell, navigating normally
    e = success           reached goal B
    f = unsafe_entry      entered a truly non-drivable cell

Observation format:
    phys (10):    x, y, heading_idx, visual_ahead_idx,
                  traffic_pass_bin, traffic_avoid_bin, traffic_slow_bin,
                  safe, legal, reached_B
    beliefs (6):  eb_water_visible, eb_maintenance_sign, eb_cones_visible,
                  eb_cars_pass, eb_cars_avoid, eb_cars_slow
    b_impl (7):   Box(drivable), Box(not_drivable), Box(shallow_puddle),
                  Box(lane_closed), Box(caution_drivable),
                  Box(road_continues), Box(blocked)   -- BEQRM only
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

from epistemic_logic import CommonGround, EpistemicState
from reward_machine import RewardMachine, RMState


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

ROAD        = 0  # drivable, clear visual
PUDDLE      = 1  # drivable, water_like visual (shallow water)
FLOODED     = 2  # NOT drivable, water_like visual (deep flood)
CONSTRUCTION = 3 # drivable (slow), cones visual
CLOSED      = 4  # NOT drivable, cones visual
DESTINATION = 5  # drivable, clear visual

CELL_DRIVABLE: Dict[int, bool] = {
    ROAD: True, PUDDLE: True, FLOODED: False,
    CONSTRUCTION: True, CLOSED: False, DESTINATION: True,
}
CELL_VISUAL: Dict[int, str] = {
    ROAD: "clear", PUDDLE: "water_like", FLOODED: "water_like",
    CONSTRUCTION: "cones", CLOSED: "cones", DESTINATION: "clear",
}
VISUAL_IDX: Dict[str, int] = {"clear": 0, "water_like": 1, "cones": 2}

# 7x7 grid; GRID[y-1][x-1] gives cell type (y=1 bottom, y=7 top)
GRID = [
    [ROAD, ROAD,    ROAD,    CONSTRUCTION, ROAD, ROAD,   DESTINATION],  # y=1
    [ROAD, FLOODED, ROAD,    ROAD,         ROAD, ROAD,   ROAD],          # y=2
    [ROAD, ROAD,    ROAD,    ROAD,         ROAD, CLOSED, ROAD],          # y=3
    [ROAD, ROAD,    PUDDLE,  ROAD,         ROAD, ROAD,   ROAD],          # y=4
    [ROAD, ROAD,    ROAD,    ROAD,    CONSTRUCTION, ROAD, ROAD],         # y=5
    [ROAD, ROAD,    ROAD,    FLOODED,      ROAD, ROAD,   ROAD],          # y=6
    [ROAD, PUDDLE,  ROAD,    ROAD,         ROAD, ROAD,   ROAD],          # y=7
]

GRID_SIZE = 7
START = (1, 7)
GOAL  = (7, 1)

N_TRAFFIC_CARS = 8
WARMUP_STEPS   = 100

# ---------------------------------------------------------------------------
# Actions and headings
# ---------------------------------------------------------------------------

ALL_ACTIONS = ["move_forward", "turn_left", "turn_right", "wait", "slow_forward", "reroute"]

HEADINGS     = ["north", "south", "east", "west"]
HEADING_DELTA = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
TURN_LEFT  = {"north": "west",  "west": "south", "south": "east", "east": "north"}
TURN_RIGHT = {"north": "east",  "east": "south", "south": "west", "west": "north"}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def get_cell(x: int, y: int) -> int:
    if 1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE:
        return GRID[y - 1][x - 1]
    return -1


def cell_is_drivable(x: int, y: int) -> bool:
    ct = get_cell(x, y)
    return ct >= 0 and CELL_DRIVABLE.get(ct, False)


# ---------------------------------------------------------------------------
# Traffic car
# ---------------------------------------------------------------------------

class TrafficCar:
    """A traffic car that knows true drivability and generates evidence."""

    def __init__(self, x: int, y: int, rng: random.Random):
        self.x = x
        self.y = y
        self.rng = rng

    def step(self, evidence: Dict[Tuple[int, int], Dict[str, int]]) -> None:
        x, y = self.x, self.y
        ct = get_cell(x, y)
        if ct == CONSTRUCTION:
            evidence[(x, y)]["cars_slow"] += 1

        dirs = list(HEADING_DELTA.values())
        self.rng.shuffle(dirs)

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if not (1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE):
                continue
            nct = get_cell(nx, ny)
            if CELL_DRIVABLE.get(nct, False):
                evidence[(nx, ny)]["cars_pass"] += 1
                self.x, self.y = nx, ny
                return
            else:
                evidence[(nx, ny)]["cars_avoid"] += 1

        # Completely boxed in — count self
        evidence[(x, y)]["cars_pass"] += 1


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AutonomousDrivingEnv:
    """Waymo-style urban driving env with traffic-based epistemic evidence."""

    def __init__(
        self,
        max_steps: int = 150,
        initial_heading: str = "east",
        step_penalty: float = 0.0,
        seed: int = 42,
    ):
        self.max_steps = max_steps
        self.initial_heading = initial_heading
        self.step_penalty = step_penalty
        self.rng = random.Random(seed)

        self.common_ground = CommonGround()
        self.state: Optional[EpistemicState] = None
        self.reward_machine = RewardMachine()
        self.done = False

        # AV mutable state
        self.pos: Tuple[int, int] = START
        self.heading: str = initial_heading
        self.step_count: int = 0
        self.left_start: bool = False
        self.prev_cell_ambiguous: bool = False
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.last_action: Optional[str] = None

        # Traffic evidence (persists across episodes; grows over training)
        self.evidence: Dict[Tuple[int, int], Dict[str, int]] = {}
        self.traffic_cars: List[TrafficCar] = []
        self._init_evidence()
        self._init_traffic()
        self._warmup()

    # ------------------------------------------------------------------
    # Evidence and traffic initialisation
    # ------------------------------------------------------------------

    def _init_evidence(self) -> None:
        for y in range(1, GRID_SIZE + 1):
            for x in range(1, GRID_SIZE + 1):
                self.evidence[(x, y)] = {"cars_pass": 0, "cars_avoid": 0, "cars_slow": 0}

    def _init_traffic(self) -> None:
        drivable = [
            (x, y)
            for y in range(1, GRID_SIZE + 1)
            for x in range(1, GRID_SIZE + 1)
            if cell_is_drivable(x, y) and (x, y) != START
        ]
        positions = self.rng.sample(drivable, min(N_TRAFFIC_CARS, len(drivable)))
        self.traffic_cars = [TrafficCar(px, py, self.rng) for px, py in positions]

    def _warmup(self) -> None:
        """Pre-populate evidence so b_impl is informative from episode 1."""
        for _ in range(WARMUP_STEPS):
            for car in self.traffic_cars:
                car.step(self.evidence)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[EpistemicState, RMState]:
        self.step_count = 0
        self.done = False
        self.reward_machine.reset()

        self.pos = START
        self.heading = self.initial_heading
        self.left_start = False
        self.prev_cell_ambiguous = False
        self.visited_cells = set()
        self.last_action = None

        # Reposition traffic cars; evidence persists and grows
        self._init_traffic()

        self.state = self._build_epistemic_state()
        return self.state, self.reward_machine.get_state()

    def get_available_actions(self) -> List[str]:
        return list(ALL_ACTIONS)

    def step(self, action: str) -> Tuple[EpistemicState, RMState, float, bool, str]:
        if self.done:
            raise RuntimeError("Episode done — call reset()")

        self.step_count += 1
        self.last_action = action

        self._apply_action(action)

        # Traffic accumulates evidence every env step
        for car in self.traffic_cars:
            car.step(self.evidence)

        self.state = self._build_epistemic_state()
        true_props = self.get_events()
        reward, rm_terminal, rm_info = self.reward_machine.step(true_props)
        reward -= self.step_penalty

        self.done = self.step_count >= self.max_steps or rm_terminal
        return self.state, self.reward_machine.get_state(), reward, self.done, rm_info

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(self, action: str) -> None:
        if action in ("move_forward", "slow_forward"):
            self._do_move()
        elif action == "turn_left":
            self.heading = TURN_LEFT[self.heading]
        elif action == "turn_right":
            self.heading = TURN_RIGHT[self.heading]
        elif action in ("wait", "reroute"):
            pass  # no-op: AV stays in place
        else:
            raise ValueError(f"Unknown action: {action}")

    def _do_move(self) -> None:
        x, y = self.pos
        dx, dy = HEADING_DELTA[self.heading]
        nx, ny = x + dx, y + dy

        if not (1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE):
            return  # boundary: no-op

        old_ct = get_cell(x, y)
        old_visual = CELL_VISUAL.get(old_ct, "clear")
        self.prev_cell_ambiguous = (old_visual != "clear")

        self.visited_cells.add((x, y))
        self.pos = (nx, ny)

        if (x, y) == START:
            self.left_start = True

    # ------------------------------------------------------------------
    # Epistemic state construction
    # ------------------------------------------------------------------

    def _next_cell(self) -> Tuple[int, int]:
        x, y = self.pos
        dx, dy = HEADING_DELTA[self.heading]
        return x + dx, y + dy

    def _build_epistemic_state(self) -> EpistemicState:
        """Build EpistemicState with explicit-belief fluents about the next cell."""
        fluents: Set[str] = set()

        x, y = self.pos
        fluents.add(f"av_{x}_{y}")
        fluents.add(f"heading_{self.heading}")
        if self.left_start:
            fluents.add("left_A")
        if (x, y) == GOAL:
            fluents.add("reached_B")

        # Beliefs about the cell ahead (what the AV must decide to enter)
        nx, ny = self._next_cell()
        if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
            visual = CELL_VISUAL[get_cell(nx, ny)]
            ev = self.evidence[(nx, ny)]

            if visual == "water_like":
                fluents.add("eb_water_visible")
            if visual == "cones":
                fluents.add("eb_cones_visible")
            # maintenance_sign is not used in this env; leave it unset

            if ev["cars_pass"] > 0:
                fluents.add("eb_cars_pass")
            else:
                fluents.add("eb_not_cars_pass")

            if ev["cars_avoid"] > 0:
                fluents.add("eb_cars_avoid")
            else:
                fluents.add("eb_not_cars_avoid")

            if ev["cars_slow"] > 0:
                fluents.add("eb_cars_slow")
            else:
                fluents.add("eb_not_cars_slow")

        return EpistemicState(frozenset(fluents), self.common_ground)

    # ------------------------------------------------------------------
    # RM event emission
    # ------------------------------------------------------------------

    def get_events(self) -> str:
        """Emit single-char RM event labels for the current timestep.

        Events are about the AV's CURRENT position after the action.
        Labels are evaluated simultaneously; the RM cascade handles ordering.
        """
        x, y = self.pos
        ct = get_cell(x, y)
        visual = CELL_VISUAL.get(ct, "clear")
        drivable = CELL_DRIVABLE.get(ct, False)
        is_ambiguous = (visual != "clear")
        is_goal = ((x, y) == GOAL)

        events = ""

        # 'a': left start cell (persistent once set)
        if self.left_start:
            events += "a"

        # 'f': at a non-drivable cell — terminal unsafe event; skip rest
        if not drivable:
            events += "f"
            return events

        # 'b': arrived at an ambiguous-visual but drivable cell
        if is_ambiguous:
            events += "b"
        else:
            # Clear cell: navigating normally
            events += "d"
            # 'c': just resolved an ambiguous cell (last cell was ambiguous)
            if self.prev_cell_ambiguous:
                events += "c"

        # 'e': reached goal
        if is_goal:
            events += "e"

        return events

    # ------------------------------------------------------------------
    # Observation accessors
    # ------------------------------------------------------------------

    def get_env_observation(self) -> Tuple[tuple, tuple]:
        """(phys, beliefs) — shared by all methods.

        phys (10):    x, y, heading_idx, visual_ahead_idx,
                      traffic_pass_bin, traffic_avoid_bin, traffic_slow_bin,
                      safe, legal, reached_B
        beliefs (6):  eb_water_visible, eb_maintenance_sign, eb_cones_visible,
                      eb_cars_pass, eb_cars_avoid, eb_cars_slow
        """
        x, y = self.pos
        heading_idx = HEADINGS.index(self.heading)

        nx, ny = self._next_cell()
        if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
            nct = get_cell(nx, ny)
            visual_ahead_idx = VISUAL_IDX[CELL_VISUAL[nct]]
            ev = self.evidence[(nx, ny)]
            traffic_pass  = int(ev["cars_pass"]  > 0)
            traffic_avoid = int(ev["cars_avoid"] > 0)
            traffic_slow  = int(ev["cars_slow"]  > 0)
        else:
            visual_ahead_idx = VISUAL_IDX["clear"]
            traffic_pass = traffic_avoid = traffic_slow = 0

        ct = get_cell(x, y)
        safe     = int(CELL_DRIVABLE.get(ct, False))
        legal    = 1  # AV is always legal if still active (UF terminates episode)
        reached_b = int((x, y) == GOAL)

        phys = (x, y, heading_idx, visual_ahead_idx,
                traffic_pass, traffic_avoid, traffic_slow,
                safe, legal, reached_b)

        s = self.state
        beliefs = (
            int(s.has("eb_water_visible"))   if s else 0,
            0,                                # eb_maintenance_sign (unused here)
            int(s.has("eb_cones_visible"))   if s else 0,
            int(s.has("eb_cars_pass"))       if s else 0,
            int(s.has("eb_cars_avoid"))      if s else 0,
            int(s.has("eb_cars_slow"))       if s else 0,
        )
        return phys, beliefs

    def get_implicit_belief_features(self) -> Tuple[int, ...]:
        """7-bit b_impl from SAT entailment over CG + explicit beliefs.

        Order: (Box(drivable), Box(not_drivable), Box(shallow_puddle),
                Box(lane_closed), Box(caution_drivable),
                Box(road_continues), Box(blocked))
        """
        if self.state is None:
            return (0,) * 7
        return tuple(int(v) for v in self.state.get_b_impl())

    def get_observation(self) -> Tuple[int, tuple, tuple]:
        phys, beliefs = self.get_env_observation()
        return self.reward_machine.get_state_index(), phys, beliefs

    def get_compact_state_str(self) -> str:
        x, y = self.pos
        ct = get_cell(x, y)
        visual = CELL_VISUAL.get(ct, "?")
        rm = self.reward_machine.get_state().value
        nx, ny = self._next_cell()
        next_v = CELL_VISUAL.get(get_cell(nx, ny), "?") if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE else "oob"
        b = self.get_implicit_belief_features()
        return (
            f"AV=({x},{y}) hdg={self.heading} visual={visual} "
            f"next=({nx},{ny}:{next_v}) RM={rm} "
            f"b_impl={b} step={self.step_count}"
        )
