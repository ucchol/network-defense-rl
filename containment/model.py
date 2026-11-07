"""Small, exact belief model for an analyst-assisted containment exercise."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np

HOSTS = ("workstation", "vpn", "jump", "app", "backups")
ZONES = ("access", "access", "operations", "operations", "recovery")
VALUES = np.array([1.0, 2.0, 3.0, 6.0, 10.0])
STATES = np.arange(1 << len(HOSTS))
PRESENT = (STATES[:, None] & (1 << np.arange(len(HOSTS)))) != 0
LOSS = PRESENT @ VALUES / VALUES.sum()
CRITICAL = PRESENT[:, 3:].any(axis=1)
TICKS = 6
INITIAL_BUDGET = 4
PRIOR = np.zeros(len(STATES))
PRIOR[[0, 1, 2]] = [.30, .50, .20]
MONITOR = (0, 2, 1, 2, 3, 4, 3)
BASE_ATTACK = ((0, 2), (1, 2), (2, 3), (2, 4), (3, 4))
ALT_ATTACK = BASE_ATTACK + ((1, 3), (0, 3))
FLAT_ATTACK = tuple(permutations(range(len(HOSTS)), 2))
SERVICE_EDGES = ((1, 2), (2, 3), (3, 4))
SERVICE_FLOWS = ((1, 3), (3, 4))
BLOCKABLE = tuple(sorted(set(ALT_ATTACK) | set(SERVICE_EDGES)))
ACTION_COST = {"wait": 0, "inspect": 1, "restrict": 1, "isolate": 2}
TOKEN_PENALTY = .018
OUTAGE_PENALTY = .22


@dataclass(frozen=True)
class Regime:
    graph: bool = True
    noisy: bool = True
    protect: bool = True
    alternate: bool = False

    @property
    def name(self) -> str:
        name = f"{'graph' if self.graph else 'flat'}"
        name += f"-{'noisy' if self.noisy else 'full'}"
        name += f"-{'protected' if self.protect else 'relaxed'}"
        return name + ("-alternate" if self.alternate else "")


@dataclass(frozen=True)
class Action:
    kind: str
    host: int = -1
    destination: int = -1

    @property
    def cost(self) -> int:
        return ACTION_COST[self.kind]

    @property
    def label(self) -> str:
        if self.kind == "wait":
            return "Wait"
        if self.kind == "restrict":
            return f"Restrict {HOSTS[self.host]} → {HOSTS[self.destination]}"
        return f"{self.kind.title()} {HOSTS[self.host]}"


WAIT = Action("wait")
ACTIONS = (
    WAIT,
    *(Action("inspect", host) for host in range(len(HOSTS))),
    *(Action("restrict", *edge) for edge in BLOCKABLE),
    *(Action("isolate", host) for host in range(len(HOSTS))),
)


@dataclass(frozen=True)
class Controls:
    budget: int = INITIAL_BUDGET
    isolated: int = 0
    blocked: frozenset[tuple[int, int]] = frozenset()

    def __post_init__(self):
        if self.budget < 0:
            raise ValueError("The response budget cannot be negative")

    def contains(self, host: int) -> bool:
        return bool(self.isolated & (1 << host))


def open_edges(edges, controls: Controls):
    return tuple(
        (src, dst) for src, dst in edges
        if not controls.contains(src) and not controls.contains(dst)
        and (src, dst) not in controls.blocked
    )


def attack_edges(regime: Regime, controls: Controls):
    edges = (ALT_ATTACK if regime.alternate else BASE_ATTACK) if regime.graph else FLAT_ATTACK
    return open_edges(edges, controls)


def can_reach(edges, source, target):
    reached = {source}
    while True:
        grown = reached | {dst for src, dst in edges if src in reached}
        if grown == reached:
            return target in reached
        reached = grown


@lru_cache(maxsize=1024)
def available_services(controls: Controls) -> tuple[bool, ...]:
    edges = open_edges(SERVICE_EDGES, controls)
    return tuple(can_reach(edges, source, target) for source, target in SERVICE_FLOWS)


@lru_cache(maxsize=4096)
def legal_moves(regime: Regime, controls: Controls) -> tuple[tuple[Action, Controls], ...]:
    open_links = set(attack_edges(regime, controls)) | set(open_edges(SERVICE_EDGES, controls))
    moves = []
    for action in ACTIONS:
        if action.cost > controls.budget:
            continue
        if action.kind == "inspect" and (not regime.noisy or controls.contains(action.host)):
            continue
        if action.kind == "isolate" and controls.contains(action.host):
            continue
        edge = action.host, action.destination
        if action.kind == "restrict" and edge not in open_links:
            continue
        isolated = controls.isolated
        blocked = controls.blocked
        if action.kind == "isolate":
            isolated |= 1 << action.host
        elif action.kind == "restrict":
            blocked = blocked | {edge}
        after = Controls(controls.budget - action.cost, isolated, blocked)
        if not regime.protect or all(available_services(after)):
            moves.append((action, after))
    return tuple(moves)


def frontier(state: int, edges) -> tuple[int, ...]:
    return tuple(sorted({
        dst for src, dst in edges
        if state & (1 << src) and not state & (1 << dst)
    }))


@lru_cache(maxsize=4096)
def transition(regime: Regime, controls: Controls, stress: bool = False) -> np.ndarray:
    """One opportunity per tick, with a possible no-spread outcome."""
    edges = attack_edges(regime, controls)
    matrix = np.zeros((len(STATES), len(STATES)))
    for state in STATES:
        targets = frontier(state, edges)
        if not targets:
            matrix[state, state] = 1.
            continue
        weights = VALUES[list(targets)] ** (2 if stress else .5)
        spread = .72 if stress else .56
        matrix[state, state] = 1. - spread
        matrix[state, [state | (1 << target) for target in targets]] = (
            spread * weights / weights.sum()
        )
    matrix.setflags(write=False)
    return matrix


def positive_probability(host: int, inspection: bool = False) -> np.ndarray:
    if inspection:
        return np.where(PRESENT[:, host], .95, .05)
    return np.where(PRESENT[:, host], .76, .13)


def initial_belief(alert: bool) -> np.ndarray:
    likelihood = positive_probability(MONITOR[0])
    weighted = PRIOR * (likelihood if alert else 1 - likelihood)
    return weighted / weighted.sum()


def observation_branches(predicted, regime: Regime, tick: int, action: Action):
    """Yield (observation, mass, posterior) without seeing the simulator state."""
    if not regime.noisy:
        for state in np.flatnonzero(predicted):
            posterior = np.zeros_like(predicted)
            posterior[state] = 1.
            yield int(state), float(predicted[state]), posterior
        return

    passive = positive_probability(MONITOR[tick])
    inspected = positive_probability(action.host, inspection=True) if action.kind == "inspect" else None
    for alert in (False, True):
        weighted = predicted * (passive if alert else 1 - passive)
        for scan in ((False, True) if inspected is not None else (None,)):
            joint = weighted
            if inspected is not None:
                joint = weighted * (inspected if scan else 1 - inspected)
            mass = float(joint.sum())
            if mass > 0:
                yield (alert, scan), mass, joint / mass


def tick_cost(predicted: np.ndarray, action: Action, after: Controls) -> float:
    unavailable = len(SERVICE_FLOWS) - sum(available_services(after))
    return float(predicted @ LOSS) + TOKEN_PENALTY * action.cost + OUTAGE_PENALTY * unavailable


def sample(probabilities, draw: float) -> int:
    cumulative = np.cumsum(probabilities)
    cumulative[np.flatnonzero(probabilities)[-1]:] = 1.
    return int(np.searchsorted(cumulative, draw, side="right"))
