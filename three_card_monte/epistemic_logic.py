"""
Epistemic Logic Engine — SAT-based reasoning for explicit and implicit beliefs.

Based on the BB-POMDP framework (Engesser et al., AAMAS 2025).

Architecture mirrors the Rust `bbases` crate:
  - State = frozenset of true fluents (atoms + explicit beliefs)
  - Explicit beliefs (Δφ) are stored directly as fluents
  - Implicit beliefs (□φ) are computed on-demand via SAT entailment:
        □φ  iff  CommonGround ∧ ExplicitBeliefs ∧ ¬φ  is UNSAT

We implement a lightweight SAT solver (DPLL with unit propagation) sufficient
for the small propositional theories in the Chop Cup domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from itertools import product


# ---------------------------------------------------------------------------
# Propositional formulas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Atom:
    """A propositional atom (e.g., 'reg_under', 'cup_down')."""
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ExplicitBelief:
    """Explicit belief fluent Δ(φ) — stored in the state as a fluent."""
    content: str  # the literal being believed, e.g. 'reg_under' or '!reg_table'

    def __repr__(self) -> str:
        return f"Δ({self.content})"


# A Fluent is either an Atom or an ExplicitBelief
Fluent = Atom | ExplicitBelief


# ---------------------------------------------------------------------------
# CNF representation for SAT solving
# ---------------------------------------------------------------------------

# A literal is (variable_name, is_positive)
Literal = Tuple[str, bool]
Clause = FrozenSet[Literal]
CNF = List[Clause]


def make_literal(var: str, positive: bool = True) -> Literal:
    return (var, positive)


def negate_literal(lit: Literal) -> Literal:
    return (lit[0], not lit[1])


# ---------------------------------------------------------------------------
# Lightweight DPLL SAT Solver
# ---------------------------------------------------------------------------

def unit_propagate(clauses: List[Set[Literal]], assignment: Dict[str, bool]) -> Optional[List[Set[Literal]]]:
    """Apply unit propagation. Returns simplified clauses or None if conflict."""
    changed = True
    while changed:
        changed = False
        unit_clauses = [c for c in clauses if len(c) == 1]
        for uc in unit_clauses:
            lit = next(iter(uc))
            var, pos = lit
            if var in assignment:
                if assignment[var] != pos:
                    return None  # conflict
                continue
            assignment[var] = pos
            changed = True
            new_clauses = []
            for c in clauses:
                if lit in c:
                    continue  # clause satisfied
                neg = negate_literal(lit)
                if neg in c:
                    new_c = c - {neg}
                    if len(new_c) == 0:
                        return None  # empty clause = conflict
                    new_clauses.append(new_c)
                else:
                    new_clauses.append(c)
            clauses = new_clauses
    return clauses


def dpll(clauses: List[Set[Literal]], assignment: Dict[str, bool]) -> bool:
    """DPLL SAT solver. Returns True if satisfiable."""
    # Unit propagation
    result = unit_propagate(clauses, assignment)
    if result is None:
        return False
    clauses = result

    if not clauses:
        return True  # all clauses satisfied

    # Pure literal elimination
    all_lits: Dict[str, Set[bool]] = {}
    for c in clauses:
        for var, pos in c:
            if var not in assignment:
                all_lits.setdefault(var, set()).add(pos)

    if not all_lits:
        # All variables assigned but clauses remain — check if any clause is unsatisfied
        for c in clauses:
            satisfied = False
            for var, pos in c:
                if var in assignment and assignment[var] == pos:
                    satisfied = True
                    break
            if not satisfied:
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

    # Choose a variable to branch on
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
    """Check if a CNF formula is satisfiable."""
    if not cnf:
        return True
    return dpll([set(c) for c in cnf], {})


# ---------------------------------------------------------------------------
# Common Ground — background knowledge shared by magician and observer
# ---------------------------------------------------------------------------

class CommonGround:
    """Encodes the Observer's background knowledge as CNF clauses.

    For the Chop Cup: the regular ball is in exactly one of {under, table, hand}.
    The Observer does NOT know about the magnetic ball.
    """

    def __init__(self, variables: List[str], exactly_one_groups: List[List[str]]):
        """
        Args:
            variables: All propositional variables in the Observer's model.
            exactly_one_groups: Groups where exactly one variable must be true.
        """
        self.variables = variables
        self.clauses: List[Set[Literal]] = []

        for group in exactly_one_groups:
            # At least one: (x1 ∨ x2 ∨ ... ∨ xn)
            self.clauses.append({make_literal(v, True) for v in group})
            # At most one: ¬xi ∨ ¬xj for all pairs
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    self.clauses.append({
                        make_literal(group[i], False),
                        make_literal(group[j], False),
                    })

    def get_clauses(self) -> List[Set[Literal]]:
        return [set(c) for c in self.clauses]


# ---------------------------------------------------------------------------
# Epistemic State — the full state with physical facts + explicit beliefs
# ---------------------------------------------------------------------------

class EpistemicState:
    """A state in the epistemic RL environment.

    Contains:
      - Physical fluents (atoms): the actual state of the world
      - Explicit beliefs (Δ): what the Observer directly observed
      - Reward machine state (tracked separately)

    Implicit beliefs (□) are NOT stored — they are computed on-demand via SAT.
    """

    def __init__(self, fluents: FrozenSet[str], common_ground: CommonGround):
        self.fluents = fluents  # set of true fluent names
        self.common_ground = common_ground
        # Cache for implicit belief queries
        self._implicit_cache: Dict[str, bool] = {}

    def copy(self) -> "EpistemicState":
        new = EpistemicState(self.fluents, self.common_ground)
        return new

    def has(self, fluent: str) -> bool:
        """Check if a fluent is true in this state."""
        return fluent in self.fluents

    def set_fluents(self, add: Set[str] | None = None, remove: Set[str] | None = None) -> "EpistemicState":
        """Return a new state with fluents added/removed."""
        new_fluents = set(self.fluents)
        if remove:
            new_fluents -= remove
        if add:
            new_fluents |= add
        return EpistemicState(frozenset(new_fluents), self.common_ground)

    def get_explicit_beliefs(self) -> Dict[str, bool]:
        """Extract explicit beliefs from the state.

        Returns a dict mapping proposition names to their believed truth values.
        E.g., {'reg_under': True, '!reg_table': True} means:
            Δ(reg_under) = true, Δ(!reg_table) = true
        """
        beliefs: Dict[str, bool] = {}
        for f in self.fluents:
            if f.startswith("eb_not_"):
                # eb_not_reg_table -> believes reg_table is false
                prop = f[7:]
                beliefs[prop] = False
            elif f.startswith("eb_"):
                # eb_reg_under -> believes reg_under is true
                prop = f[3:]
                beliefs[prop] = True
        return beliefs

    def get_explicit_belief_clauses(self) -> List[Set[Literal]]:
        """Convert explicit beliefs to SAT clauses for entailment checking."""
        clauses: List[Set[Literal]] = []
        for f in self.fluents:
            if f.startswith("eb_not_"):
                # Δ(¬p): observer believes p is false => unit clause ¬p
                prop = f[7:]
                clauses.append({make_literal(prop, False)})
            elif f.startswith("eb_"):
                # Δ(p): observer believes p is true => unit clause p
                prop = f[3:]
                clauses.append({make_literal(prop, True)})
        return clauses

    def implicit_belief(self, proposition: str, positive: bool = True) -> bool:
        """Compute implicit belief □φ (or □¬φ) via SAT entailment.

        □φ iff CommonGround ∧ ExplicitBeliefs ∧ ¬φ is UNSAT

        This is the core epistemic reasoning mechanism — the Observer
        DEDUCES what must be true from background knowledge + observations.
        """
        cache_key = f"{proposition}:{positive}"
        if cache_key in self._implicit_cache:
            return self._implicit_cache[cache_key]

        # Build KB = CommonGround ∧ ExplicitBeliefs
        kb_clauses: List[Set[Literal]] = []
        kb_clauses.extend(self.common_ground.get_clauses())
        kb_clauses.extend(self.get_explicit_belief_clauses())

        # Add ¬φ (negation of what we're querying)
        # If checking □φ (positive=True): add ¬φ, check UNSAT
        # If checking □¬φ (positive=False): add φ, check UNSAT
        query_lit = make_literal(proposition, not positive)
        kb_clauses.append({query_lit})

        # □φ iff KB ∧ ¬φ is UNSAT
        result = not is_satisfiable(kb_clauses)
        self._implicit_cache[cache_key] = result
        return result

    def as_hashable(self) -> FrozenSet[str]:
        """Return a hashable representation for Q-table keys."""
        return self.fluents

    def __eq__(self, other) -> bool:
        if not isinstance(other, EpistemicState):
            return False
        return self.fluents == other.fluents

    def __hash__(self) -> int:
        return hash(self.fluents)

    def __repr__(self) -> str:
        physical = sorted(f for f in self.fluents if not f.startswith("eb_") and not f.startswith("rm_"))
        beliefs = sorted(f for f in self.fluents if f.startswith("eb_"))
        rm = sorted(f for f in self.fluents if f.startswith("rm_"))
        return f"State(phys={physical}, beliefs={beliefs}, rm={rm})"
