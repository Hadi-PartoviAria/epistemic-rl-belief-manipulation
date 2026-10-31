"""Autonomous Driving Environment -- ablation variant (7x7 grid, randomised cell types).

Identical to the main 7x7 Waymo env EXCEPT:

  1. RANDOMIZABLE_WATER_CELLS are randomly assigned as PUDDLE or FLOODED at
     the start of every episode (via _randomize_grid / reset).  Tabular
     memorisation of cell-specific drivability is therefore impossible;
     agents must use per-episode evidence to infer cell types.

  2. WARMUP_STEPS = 50 (run on every reset) to populate traffic evidence for
     the new cell assignment before the AV moves.

  3. get_cell() is an instance method that reads the episode-specific grid
     rather than a fixed module-level constant.

These changes are *ablation-only*; the main comparison env (autonomous_driving_6x6/)
is unchanged.

The design is intended to show that QRM (which strips traffic-count evidence
from its obs, see ablation_agent.py) cannot infer per-episode drivability and
therefore fails on randomised water cells, while BEQRM's SAT-derived b_impl
correctly resolves drivability each episode.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Set, Tuple

from epistemic_logic import CommonGround, EpistemicState
from reward_machine import RewardMachine, RMState


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

ROAD         = 0   # drivable, clear visual
PUDDLE       = 1   # drivable, water_like visual (shallow water)
FLOODED      = 2   # NOT drivable, water_like visual (deep flood)
CONSTRUCTION = 3   # drivable (slow), cones visual
CLOSED       = 4   # NOT drivable, cones visual
DESTINATION  = 5   # drivable, clear visual

CELL_DRIVABLE: Dict[int, bool] = {
    ROAD: True, PUDDLE: True, FLOODED: False,
    CONSTRUCTION: True, CLOSED: False, DESTINATION: True,
}
CELL_VISUAL: Dict[int, str] = {
    ROAD: "clear", PUDDLE: "water_like", FLOODED: "water_like",
    CONSTRUCTION: "cones", CLOSED: "cones", DESTINATION: "clear",
}
VISUAL_IDX: Dict[str, int] = {"clear": 0, "water_like": 1, "cones": 2}

# Base grid template (y=1 bottom, y=7 top; index as BASE_GRID[y-1][x-1]).
# RANDOMIZABLE_WATER_CELLS positions will be overridden each episode.
BASE_GRID = [
    [ROAD, ROAD,    ROAD,    CONSTRUCTION, ROAD, ROAD,   DESTINATION],  # y=1
    [ROAD, FLOODED, ROAD,    ROAD,         ROAD, ROAD,   ROAD],          # y=2
    [ROAD, ROAD,    ROAD,    ROAD,         ROAD, CLOSED, ROAD],          # y=3
    [ROAD, ROAD,    PUDDLE,  ROAD,         ROAD, ROAD,   ROAD],          # y=4
    [ROAD, ROAD,    ROAD,    ROAD,    CONSTRUCTION, ROAD, ROAD],         # y=5
    [ROAD, ROAD,    ROAD,    FLOODED,      ROAD, ROAD,   ROAD],          # y=6
    [ROAD, PUDDLE,  ROAD,    ROAD,         ROAD, ROAD,   ROAD],          # y=7
]

# These four positions are randomly reassigned as PUDDLE or FLOODED each episode.
RANDOMIZABLE_WATER_CELLS: List[Tuple[int, int]] = [(2, 7), (4, 6), (3, 4), (2, 2)]

GRID_SIZE = 7
START = (1, 7)
GOAL  = (7, 1)

N_TRAFFIC_CARS = 8
WARMUP_STEPS   = 50   # run on every reset for the new cell assignment

# ---------------------------------------------------------------------------
# Actions and headings
# ---------------------------------------------------------------------------

ALL_ACTIONS = ["move_forward", "turn_left", "turn_right", "wait", "slow_forward", "reroute"]

HEADINGS      = ["north", "south", "east", "west"]
HEADING_DELTA = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
TURN_LEFT  = {"north": "west",  "west": "south", "south": "east", "east": "north"}
TURN_RIGHT = {"north": "east",  "east": "south", "south": "west", "west": "north"}


# ---------------------------------------------------------------------------
# Traffic car
# ---------------------------------------------------------------------------

class TrafficCar:
    """A traffic car that knows true drivability and generates evidence.

    Uses a passed-in cell_fn so it respects the episode-specific cell
    assignment rather than a fixed module-level grid.
    """

    def __init__(self, x: int, y: int, rng: random.Random,
                 cell_fn: Callable[[int, int], int]):
        self.x = x
        self.y = y
        self.rng = rng
        self.cell_fn = cell_fn

    def step(self, evidence: Dict[Tuple[int, int], Dict[str, int]]) -> None:
        x, y = self.x, self.y
        ct = self.cell_fn(x, y)
        if ct == CONSTRUCTION:
            evidence[(x, y)]["cars_slow"] += 1

        dirs = list(HEADING_DELTA.values())
        self.rng.shuffle(dirs)

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if not (1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE):
                continue
            nct = self.cell_fn(nx, ny)
            if CELL_DRIVABLE.get(nct, False):
                evidence[(nx, ny)]["cars_pass"] += 1
                self.x, self.y = nx, ny
                return
            else:
                evidence[(nx, ny)]["cars_avoid"] += 1

        evidence[(x, y)]["cars_pass"] += 1


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AutonomousDrivingEnv:
    """Waymo-style urban driving env — ablation variant with randomised cell types."""

    def __init__(
        self,
        max_steps: int = 150,
        initial_heading: str = "east",
        step_penalty: float = 0.0,
        seed: int = 42,
        randomize_cells: bool = True,
    ):
        self.max_steps = max_steps
        self.initial_heading = initial_heading
        self.step_penalty = step_penalty
        self.randomize_cells = randomize_cells
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

        # Episode-specific cell grid (randomised each reset)
        self._episode_grid: Dict[Tuple[int, int], int] = {}
        self._randomize_grid()

        # Traffic evidence and cars (reset each episode with new cell types)
        self.evidence: Dict[Tuple[int, int], Dict[str, int]] = {}
        self.traffic_cars: List[TrafficCar] = []
        self._init_evidence()
        self._init_traffic()
        self._warmup()

    # ------------------------------------------------------------------
    # Instance-level grid accessors
    # ------------------------------------------------------------------

    def get_cell(self, x: int, y: int) -> int:
        """Return the cell type for the current episode's grid assignment."""
        if 1 <= x <= GRID_SIZE and 1 <= y <= GRID_SIZE:
            return self._episode_grid.get((x, y), BASE_GRID[y - 1][x - 1])
        return -1

    def _cell_is_drivable(self, x: int, y: int) -> bool:
        ct = self.get_cell(x, y)
        return ct >= 0 and CELL_DRIVABLE.get(ct, False)

    def _randomize_grid(self) -> None:
        """Randomly assign PUDDLE or FLOODED to each randomisable water cell."""
        for pos in RANDOMIZABLE_WATER_CELLS:
            self._episode_grid[pos] = PUDDLE if self.rng.random() < 0.5 else FLOODED

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
            if self._cell_is_drivable(x, y) and (x, y) != START
        ]
        positions = self.rng.sample(drivable, min(N_TRAFFIC_CARS, len(drivable)))
        self.traffic_cars = [
            TrafficCar(px, py, self.rng, self.get_cell) for px, py in positions
        ]

    def _warmup(self) -> None:
        """Populate evidence for the current episode's cell assignment."""
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

        # New cell assignment, fresh evidence, new traffic for this episode
        self._randomize_grid()
        self._init_evidence()
        self._init_traffic()
        self._warmup()

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
            pass
        else:
            raise ValueError(f"Unknown action: {action}")

    def _do_move(self) -> None:
        x, y = self.pos
        dx, dy = HEADING_DELTA[self.heading]
        nx, ny = x + dx, y + dy

        if not (1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE):
            return

        old_ct = self.get_cell(x, y)
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
        fluents: Set[str] = set()

        x, y = self.pos
        fluents.add(f"av_{x}_{y}")
        fluents.add(f"heading_{self.heading}")
        if self.left_start:
            fluents.add("left_A")
        if (x, y) == GOAL:
            fluents.add("reached_B")

        nx, ny = self._next_cell()
        if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
            visual = CELL_VISUAL[self.get_cell(nx, ny)]
            ev = self.evidence[(nx, ny)]

            if visual == "water_like":
                fluents.add("eb_water_visible")
            if visual == "cones":
                fluents.add("eb_cones_visible")

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
        x, y = self.pos
        ct = self.get_cell(x, y)
        visual = CELL_VISUAL.get(ct, "clear")
        drivable = CELL_DRIVABLE.get(ct, False)
        is_ambiguous = (visual != "clear")
        is_goal = ((x, y) == GOAL)

        events = ""

        if self.left_start:
            events += "a"

        if not drivable:
            events += "f"
            return events

        if is_ambiguous:
            events += "b"
        else:
            events += "d"
            if self.prev_cell_ambiguous:
                events += "c"

        if is_goal:
            events += "e"

        return events

    # ------------------------------------------------------------------
    # Observation accessors
    # ------------------------------------------------------------------

    def get_env_observation(self) -> Tuple[tuple, tuple]:
        """(phys, beliefs) — phys includes traffic-count evidence bins."""
        x, y = self.pos
        heading_idx = HEADINGS.index(self.heading)

        nx, ny = self._next_cell()
        if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
            nct = self.get_cell(nx, ny)
            visual_ahead_idx = VISUAL_IDX[CELL_VISUAL[nct]]
            ev = self.evidence[(nx, ny)]
            traffic_pass  = int(ev["cars_pass"]  > 0)
            traffic_avoid = int(ev["cars_avoid"] > 0)
            traffic_slow  = int(ev["cars_slow"]  > 0)
        else:
            visual_ahead_idx = VISUAL_IDX["clear"]
            traffic_pass = traffic_avoid = traffic_slow = 0

        ct = self.get_cell(x, y)
        safe      = int(CELL_DRIVABLE.get(ct, False))
        legal     = 1
        reached_b = int((x, y) == GOAL)

        phys = (x, y, heading_idx, visual_ahead_idx,
                traffic_pass, traffic_avoid, traffic_slow,
                safe, legal, reached_b)

        s = self.state
        beliefs = (
            int(s.has("eb_water_visible"))  if s else 0,
            0,
            int(s.has("eb_cones_visible"))  if s else 0,
            int(s.has("eb_cars_pass"))      if s else 0,
            int(s.has("eb_cars_avoid"))     if s else 0,
            int(s.has("eb_cars_slow"))      if s else 0,
        )
        return phys, beliefs

    def get_implicit_belief_features(self) -> Tuple[int, ...]:
        """7-bit b_impl from SAT entailment."""
        if self.state is None:
            return (0,) * 7
        return tuple(int(v) for v in self.state.get_b_impl())

    def get_observation(self) -> Tuple[int, tuple, tuple]:
        phys, beliefs = self.get_env_observation()
        return self.reward_machine.get_state_index(), phys, beliefs

    def get_compact_state_str(self) -> str:
        x, y = self.pos
        ct = self.get_cell(x, y)
        visual = CELL_VISUAL.get(ct, "?")
        rm = self.reward_machine.get_state().value
        nx, ny = self._next_cell()
        next_v = (CELL_VISUAL.get(self.get_cell(nx, ny), "?")
                  if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE else "oob")
        b = self.get_implicit_belief_features()
        return (
            f"AV=({x},{y}) hdg={self.heading} visual={visual} "
            f"next=({nx},{ny}:{next_v}) RM={rm} "
            f"b_impl={b} step={self.step_count}"
        )
