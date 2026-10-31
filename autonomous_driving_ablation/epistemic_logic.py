"""Epistemic Logic Engine -- SAT-based reasoning for the Waymo driving task.

The Observer (auditor / traffic-management system) reasons about whether
road cells are drivable by combining:
  - Explicit beliefs (Delta phi): direct sensor observations and traffic counts
  - Common Ground (CG): shared traffic-world knowledge encoded as CNF clauses
  - Implicit beliefs (Box phi): SAT entailment over CG + explicit beliefs

Common Ground rules
-------------------
Water / puddle / flooding:
  water_visible(c) AND cars_pass(c)   -> shallow_puddle(c) -> drivable(c)
  water_visible(c) AND cars_avoid(c)  -> flooded(c)        -> not_drivable(c)

Maintenance / construction / lane closure:
  maintenance_sign(c)                 -> construction_zone(c)
  cones_visible(c)                    -> construction_zone(c)
  construction_zone(c) AND cars_avoid(c)               -> lane_closed(c)  -> not_drivable(c)
  construction_zone(c) AND cars_pass(c) AND cars_slow(c) -> caution_drivable(c) -> drivable(c)

Traffic-flow evidence:
  cars_pass(c)   -> traffic_flow(c) -> road_continues(c) -> drivable(c)
  cars_avoid(c) AND NOT cars_pass(c) -> blocked(c) -> not_drivable(c)

Mutual exclusion: not_drivable(c) AND drivable(c) is always UNSAT
  {not_drivable, drivable} represents this via the clause {-not_drivable, -drivable}.

Queried implicit beliefs (b_impl):
  Box(drivable)         -- cell ahead is confirmed safe to enter
  Box(not_drivable)     -- cell ahead is confirmed unsafe
  Box(shallow_puddle)   -- confirmed shallow puddle (drivable water)
  Box(lane_closed)      -- confirmed lane closure
  Box(caution_drivable) -- confirmed drivable but requires caution
  Box(road_continues)   -- traffic flow confirms road continuity
  Box(blocked)          -- traffic avoidance confirms blockage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


Literal = Tuple[str, bool]
Clause = FrozenSet[Literal]
CNF = List[Clause]


def make_literal(var: str, positive: bool = True) -> Literal:
    return (var, positive)


def negate_literal(lit: Literal) -> Literal:
    return (lit[0], not lit[1])


# ---------------------------------------------------------------------------
# DPLL SAT Solver (unchanged from previous version)
# ---------------------------------------------------------------------------

def unit_propagate(clauses: List[Set[Literal]],
                   assignment: Dict[str, bool]) -> Optional[List[Set[Literal]]]:
    changed = True
    while changed:
        changed = False
        unit_clauses = [c for c in clauses if len(c) == 1]
        for uc in unit_clauses:
            lit = next(iter(uc))
            var, pos = lit
            if var in assignment:
                if assignment[var] != pos:
                    return None
                continue
            assignment[var] = pos
            changed = True
            new_clauses = []
            for c in clauses:
                if lit in c:
                    continue
                neg = negate_literal(lit)
                if neg in c:
                    new_c = c - {neg}
                    if not new_c:
                        return None
                    new_clauses.append(new_c)
                else:
                    new_clauses.append(c)
            clauses = new_clauses
    return clauses


def dpll(clauses: List[Set[Literal]], assignment: Dict[str, bool]) -> bool:
    result = unit_propagate(clauses, assignment)
    if result is None:
        return False
    clauses = result
    if not clauses:
        return True

    all_lits: Dict[str, Set[bool]] = {}
    for c in clauses:
        for var, pos in c:
            if var not in assignment:
                all_lits.setdefault(var, set()).add(pos)

    if not all_lits:
        for c in clauses:
            if not any(assignment.get(var) == pos for var, pos in c):
                return False
        return True

    for var, polarities in all_lits.items():
        if len(polarities) == 1:
            pos = next(iter(polarities))
            assignment[var] = pos
            new_clauses = []
            for c in clauses:
                if (var, pos) in c:
                    continue
                neg = (var, not pos)
                if neg in c:
                    new_c = c - {neg}
                    if not new_c:
                        return False
                    new_clauses.append(new_c)
                else:
                    new_clauses.append(c)
            return dpll(new_clauses, assignment)

    branch_var = next(iter(all_lits))
    for val in [True, False]:
        new_assignment = dict(assignment)
        new_assignment[branch_var] = val
        lit = (branch_var, val)
        neg = (branch_var, not val)
        new_clauses = []
        conflict = False
        for c in clauses:
            if lit in c:
                continue
            if neg in c:
                new_c = c - {neg}
                if not new_c:
                    conflict = True
                    break
                new_clauses.append(new_c)
            else:
                new_clauses.append(c)
        if not conflict and dpll(new_clauses, new_assignment):
            assignment.update(new_assignment)
            return True
    return False


def is_satisfiable(cnf: List[Set[Literal]]) -> bool:
    if not cnf:
        return True
    return dpll([set(c) for c in cnf], {})


# ---------------------------------------------------------------------------
# Common Ground -- Waymo traffic-world knowledge
# ---------------------------------------------------------------------------

class CommonGround:
    """Traffic-world axioms for the Waymo urban driving scenario.

    All rules are physical/observable: they relate directly observed
    evidence (water, cones, traffic counts) to drivability conclusions.
    No implicit beliefs appear in the rule antecedents.
    """

    def __init__(self):
        self.clauses: List[Set[Literal]] = []
        self._build_traffic_rules()

    def _build_traffic_rules(self) -> None:
        P = make_literal   # shorthand

        # -- Water / puddle rules --
        # water_visible AND cars_pass -> shallow_puddle
        self.clauses.append({P("water_visible", False), P("cars_pass", False),
                             P("shallow_puddle", True)})
        # shallow_puddle -> drivable
        self.clauses.append({P("shallow_puddle", False), P("drivable", True)})
        # water_visible AND cars_avoid -> flooded
        self.clauses.append({P("water_visible", False), P("cars_avoid", False),
                             P("flooded", True)})
        # flooded -> not_drivable
        self.clauses.append({P("flooded", False), P("not_drivable", True)})

        # -- Maintenance / construction rules --
        # maintenance_sign -> construction_zone
        self.clauses.append({P("maintenance_sign", False),
                             P("construction_zone", True)})
        # cones_visible -> construction_zone
        self.clauses.append({P("cones_visible", False),
                             P("construction_zone", True)})
        # construction_zone AND cars_avoid -> lane_closed
        self.clauses.append({P("construction_zone", False), P("cars_avoid", False),
                             P("lane_closed", True)})
        # lane_closed -> not_drivable
        self.clauses.append({P("lane_closed", False), P("not_drivable", True)})
        # construction_zone AND cars_pass AND cars_slow -> caution_drivable
        self.clauses.append({P("construction_zone", False), P("cars_pass", False),
                             P("cars_slow", False), P("caution_drivable", True)})
        # caution_drivable -> drivable
        self.clauses.append({P("caution_drivable", False), P("drivable", True)})

        # -- Traffic-flow rules --
        # cars_pass -> traffic_flow
        self.clauses.append({P("cars_pass", False), P("traffic_flow", True)})
        # traffic_flow -> road_continues
        self.clauses.append({P("traffic_flow", False), P("road_continues", True)})
        # road_continues -> drivable
        self.clauses.append({P("road_continues", False), P("drivable", True)})
        # cars_avoid AND NOT cars_pass -> blocked
        self.clauses.append({P("cars_avoid", False), P("cars_pass", True),
                             P("blocked", True)})
        # blocked -> not_drivable
        self.clauses.append({P("blocked", False), P("not_drivable", True)})

        # -- Mutual exclusion: drivable and not_drivable cannot both hold --
        self.clauses.append({P("not_drivable", False), P("drivable", False)})

    def get_clauses(self) -> List[Set[Literal]]:
        return [set(c) for c in self.clauses]


# ---------------------------------------------------------------------------
# Epistemic State
# ---------------------------------------------------------------------------

class EpistemicState:
    """Waymo epistemic state.

    Fluents are prefixed:
      eb_<prop>      -- explicit belief: <prop> is observed true
      eb_not_<prop>  -- explicit belief: <prop> is observed false

    Implicit beliefs are computed on-demand by SAT entailment.
    """

    def __init__(self, fluents: FrozenSet[str], common_ground: CommonGround):
        self.fluents = fluents
        self.common_ground = common_ground
        self._implicit_cache: Dict[str, bool] = {}

    def copy(self) -> "EpistemicState":
        return EpistemicState(self.fluents, self.common_ground)

    def has(self, fluent: str) -> bool:
        return fluent in self.fluents

    def set_fluents(self, add: Set[str] | None = None,
                    remove: Set[str] | None = None) -> "EpistemicState":
        new_fluents = set(self.fluents)
        if remove:
            new_fluents -= remove
        if add:
            new_fluents |= add
        return EpistemicState(frozenset(new_fluents), self.common_ground)

    def get_explicit_belief_clauses(self) -> List[Set[Literal]]:
        clauses: List[Set[Literal]] = []
        for f in self.fluents:
            if f.startswith("eb_not_"):
                prop = f[7:]
                clauses.append({make_literal(prop, False)})
            elif f.startswith("eb_"):
                prop = f[3:]
                clauses.append({make_literal(prop, True)})
        return clauses

    def implicit_belief(self, proposition: str, positive: bool = True) -> bool:
        """Compute Box(phi) or Box(-phi) via SAT entailment.

        Box(phi) iff CG AND ExplicitBeliefs AND -phi is UNSAT.
        """
        cache_key = f"{proposition}:{positive}"
        if cache_key in self._implicit_cache:
            return self._implicit_cache[cache_key]

        kb: List[Set[Literal]] = []
        kb.extend(self.common_ground.get_clauses())
        kb.extend(self.get_explicit_belief_clauses())
        kb.append({make_literal(proposition, not positive)})

        result = not is_satisfiable(kb)
        self._implicit_cache[cache_key] = result
        return result

    def get_b_impl(self) -> Tuple[bool, ...]:
        """Return the 7-bit implicit-belief feature vector for BEQRM.

        Order matches the paper's b_impl definition:
          (Box(drivable), Box(not_drivable), Box(shallow_puddle),
           Box(lane_closed), Box(caution_drivable),
           Box(road_continues), Box(blocked))
        """
        return (
            self.implicit_belief("drivable",         True),
            self.implicit_belief("not_drivable",     True),
            self.implicit_belief("shallow_puddle",   True),
            self.implicit_belief("lane_closed",      True),
            self.implicit_belief("caution_drivable", True),
            self.implicit_belief("road_continues",   True),
            self.implicit_belief("blocked",          True),
        )

    def as_hashable(self) -> FrozenSet[str]:
        return self.fluents

    def __eq__(self, other) -> bool:
        return isinstance(other, EpistemicState) and self.fluents == other.fluents

    def __hash__(self) -> int:
        return hash(self.fluents)

    def __repr__(self) -> str:
        bel = sorted(f for f in self.fluents if f.startswith("eb_"))
        return f"EpistemicState(beliefs={bel})"
