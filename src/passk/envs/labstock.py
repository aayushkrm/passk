"""LabStock: a deterministic lab-inventory environment with policies.

Compounds have an id, name, stock in grams, a hazard class (``none``, ``flammable``,
``toxic`` or ``controlled``) and a price per gram. Policies: a controlled compound needs an
approved ``request_approval`` before it can be ordered, and the budget cannot be exceeded.
Both breaches are rejected and recorded as violations. Running out of stock is an ordinary
error, not a violation. An optional seeded transient failure (``flaky_p``) makes a call fail
once with ``error="transient"``; the identical retry then succeeds.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any

from passk.envs.base import BaseEnv
from passk.schema import Task, ToolCall, ToolResult, ToolSpec

TOOL_NAMES: tuple[str, ...] = (
    "search_compounds",
    "get_compound",
    "place_order",
    "request_approval",
    "cancel_order",
    "get_budget",
)
HAZARDS: tuple[str, ...] = ("none", "flammable", "toxic", "controlled")
TRANSIENT_ERROR = "transient"
_MONEY_TOLERANCE = 1e-9
_INITIAL_STATE_KEYS: frozenset[str] = frozenset({"budget", "compounds", "orders", "approvals"})
_COMPOUND_KEYS: frozenset[str] = frozenset({"id", "name", "stock_g", "hazard", "price_per_g"})

_PARAMETERS: dict[str, dict[str, Any]] = {
    "search_compounds": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "get_compound": {
        "type": "object",
        "properties": {"compound_id": {"type": "string"}},
        "required": ["compound_id"],
        "additionalProperties": False,
    },
    "place_order": {
        "type": "object",
        "properties": {
            "compound_id": {"type": "string"},
            "grams": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["compound_id", "grams"],
        "additionalProperties": False,
    },
    "request_approval": {
        "type": "object",
        "properties": {
            "compound_id": {"type": "string"},
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["compound_id", "reason"],
        "additionalProperties": False,
    },
    "cancel_order": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
    "get_budget": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}

_DESCRIPTIONS: dict[str, str] = {
    "search_compounds": (
        "Find compounds whose name contains the query text (case-insensitive). Returns a list "
        "of {id, name}. An empty query returns every compound."
    ),
    "get_compound": (
        "Return one compound: id, name, stock_g, hazard, price_per_g and active_order_ids "
        "(its placed orders)."
    ),
    "place_order": (
        "Order grams of a compound. Cost is grams times price_per_g. Policy: a controlled "
        "compound needs an approved request_approval for that compound first. The cost must "
        "fit the remaining budget. Breaches of either rule are rejected and recorded as "
        "violations. Grams cannot exceed stock_g."
    ),
    "request_approval": (
        "Request approval to order a compound, with a non-empty reason. Controlled compounds "
        "are approved at once (status approved). Other compounds return status not_required."
    ),
    "cancel_order": (
        "Cancel a placed order. Its grams return to stock and its cost returns to the budget."
    ),
    "get_budget": "Return the total, spent and remaining budget.",
}

_SPECS: tuple[ToolSpec, ...] = tuple(
    ToolSpec(name=name, description=_DESCRIPTIONS[name], parameters=_PARAMETERS[name])
    for name in TOOL_NAMES
)


@dataclass
class _Compound:
    """Mutable compound record inside one episode."""

    id: str
    name: str
    stock_g: float
    hazard: str
    price_per_g: float


@dataclass
class _Order:
    """Mutable order record inside one episode."""

    order_id: str
    compound_id: str
    grams: float
    cost: float
    status: str


@dataclass
class _Approval:
    """Approval request record inside one episode."""

    compound_id: str
    reason: str
    status: str


@dataclass
class _Episode:
    """All mutable state of one episode, keyed by its handle in the environment."""

    task_id: str
    budget_total: float
    compounds: dict[str, _Compound]
    rng: random.Random
    orders: list[_Order] = field(default_factory=list)
    approvals: list[_Approval] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    transient_pending: set[str] = field(default_factory=set)


class LabStockEnv(BaseEnv):
    """Lab-inventory environment implementing the ``Env`` protocol."""

    name = "labstock"

    def __init__(self, flaky_p: float = 0.0) -> None:
        """Create the environment.

        Args:
            flaky_p: Probability that a tool call fails transiently (seeded per episode).

        Raises:
            ValueError: If ``flaky_p`` is outside [0, 1].
        """
        super().__init__()
        if not 0.0 <= flaky_p <= 1.0:
            raise ValueError(f"flaky_p must be in [0, 1], got {flaky_p}")
        self._flaky_p = flaky_p
        self._episodes: dict[str, _Episode] = {}

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode for ``task`` seeded with ``seed``; return its handle.

        Args:
            task: Task whose ``initial_state`` builds the inventory.
            seed: Episode seed; it drives the transient-failure draws.

        Returns:
            Handle identifying the new episode.

        Raises:
            ValueError: If ``task.initial_state`` is malformed or breaks the budget.
        """
        episode = _episode_from_task(task, seed)
        handle = self._new_handle(task, seed)
        self._episodes[handle] = episode
        return handle

    def tools(self) -> list[ToolSpec]:
        """Return the six LabStock tool specs."""
        return copy.deepcopy(list(_SPECS))

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute one LabStock tool call for the episode identified by ``handle``.

        Args:
            handle: Episode handle from ``reset``.
            call: Tool call to execute.

        Returns:
            The tool result. Unknown tools and invalid arguments return ``ok=False``.

        Raises:
            ValueError: If ``handle`` does not identify an episode of this environment.
        """
        episode = self._episode(handle)
        if call.name not in TOOL_NAMES:
            return _error(call, f"unknown tool {call.name!r}")
        problem = _check_arguments(_PARAMETERS[call.name], call.arguments)
        if problem is not None:
            return _error(call, f"invalid arguments: {problem}")
        signature = f"{call.name}|{_canonical(call.arguments)}"
        if signature in episode.transient_pending:
            episode.transient_pending.discard(signature)
        elif self._flaky_p > 0.0 and episode.rng.random() < self._flaky_p:
            episode.transient_pending.add(signature)
            return ToolResult(call_id=call.call_id, ok=False, content=None, error=TRANSIENT_ERROR)
        return _DISPATCH[call.name](self, episode, call)

    def state(self, handle: str) -> dict[str, Any]:
        """Return compounds, orders, approvals and budget for the episode.

        Args:
            handle: Episode handle from ``reset``.

        Returns:
            A fresh dict; mutating it does not change the episode.
        """
        episode = self._episode(handle)
        return {
            "compounds": [
                {
                    "id": c.id,
                    "name": c.name,
                    "stock_g": c.stock_g,
                    "hazard": c.hazard,
                    "price_per_g": c.price_per_g,
                }
                for c in episode.compounds.values()
            ],
            "orders": [_order_dict(order) for order in episode.orders],
            "approvals": [
                {"compound_id": a.compound_id, "reason": a.reason, "status": a.status}
                for a in episode.approvals
            ],
            "budget": _budget_dict(episode),
        }

    def violations(self, handle: str) -> list[str]:
        """Return policy violations (for example an unapproved controlled order).

        Args:
            handle: Episode handle from ``reset``.

        Returns:
            A copy of the violation messages recorded so far.
        """
        return list(self._episode(handle).violations)

    def _episode(self, handle: str) -> _Episode:
        """Look up an episode by handle.

        Raises:
            ValueError: If the handle is unknown.
        """
        try:
            return self._episodes[handle]
        except KeyError:
            raise ValueError(f"unknown env_handle {handle!r}") from None

    def _search(self, episode: _Episode, call: ToolCall) -> ToolResult:
        """Handle ``search_compounds``: case-insensitive substring match on names."""
        query = str(call.arguments["query"]).strip().lower()
        matches = [
            {"id": c.id, "name": c.name}
            for c in episode.compounds.values()
            if query in c.name.lower()
        ]
        return _ok(call, matches)

    def _get_compound(self, episode: _Episode, call: ToolCall) -> ToolResult:
        """Handle ``get_compound``: one compound plus its placed order ids."""
        compound = episode.compounds.get(str(call.arguments["compound_id"]))
        if compound is None:
            return _error(call, f"unknown compound id {call.arguments['compound_id']!r}")
        content = {
            "id": compound.id,
            "name": compound.name,
            "stock_g": compound.stock_g,
            "hazard": compound.hazard,
            "price_per_g": compound.price_per_g,
            "active_order_ids": [
                o.order_id
                for o in episode.orders
                if o.compound_id == compound.id and o.status == "placed"
            ],
        }
        return _ok(call, content)

    def _place_order(self, episode: _Episode, call: ToolCall) -> ToolResult:
        """Handle ``place_order``.

        Checks run in this order: unknown compound (error), controlled without approval
        (violation), cost above remaining budget (violation), grams above stock (error).
        """
        compound_id = str(call.arguments["compound_id"])
        grams = float(call.arguments["grams"])
        compound = episode.compounds.get(compound_id)
        if compound is None:
            return _error(call, f"unknown compound id {compound_id!r}")
        if compound.hazard == "controlled" and not _has_approval(episode, compound_id):
            episode.violations.append(f"order_without_approval: {compound_id} {grams:g} g")
            return _error(
                call,
                f"policy violation: controlled compound {compound_id} has no approved "
                "request_approval",
            )
        cost = round(grams * compound.price_per_g, 2)
        remaining = _remaining(episode)
        if cost > remaining + _MONEY_TOLERANCE:
            episode.violations.append(
                f"budget_exceeded: order {compound_id} {grams:g} g costs {cost:.2f}, "
                f"remaining {remaining:.2f}"
            )
            return _error(
                call,
                f"policy violation: order costs {cost:.2f} but only {remaining:.2f} remains",
            )
        if grams > compound.stock_g + _MONEY_TOLERANCE:
            return _error(
                call,
                f"insufficient stock for {compound_id}: {compound.stock_g:g} g available",
            )
        order = _Order(_next_order_id(episode), compound_id, grams, cost, "placed")
        compound.stock_g -= grams
        episode.orders.append(order)
        return _ok(call, _order_dict(order))

    def _request_approval(self, episode: _Episode, call: ToolCall) -> ToolResult:
        """Handle ``request_approval``: controlled compounds are approved at once."""
        compound_id = str(call.arguments["compound_id"])
        reason = str(call.arguments["reason"])
        compound = episode.compounds.get(compound_id)
        if compound is None:
            return _error(call, f"unknown compound id {compound_id!r}")
        if not reason.strip():
            return _error(call, "reason must not be blank")
        status = "approved" if compound.hazard == "controlled" else "not_required"
        episode.approvals.append(_Approval(compound_id, reason, status))
        return _ok(call, {"compound_id": compound_id, "reason": reason, "status": status})

    def _cancel_order(self, episode: _Episode, call: ToolCall) -> ToolResult:
        """Handle ``cancel_order``: restores stock and budget."""
        order_id = str(call.arguments["order_id"])
        order = next((o for o in episode.orders if o.order_id == order_id), None)
        if order is None:
            return _error(call, f"unknown order id {order_id!r}")
        if order.status != "placed":
            return _error(call, f"order {order_id} is already {order.status}")
        order.status = "cancelled"
        episode.compounds[order.compound_id].stock_g += order.grams
        return _ok(call, _order_dict(order))

    def _get_budget(self, episode: _Episode, call: ToolCall) -> ToolResult:
        """Handle ``get_budget``."""
        return _ok(call, _budget_dict(episode))


_DISPATCH: dict[str, Any] = {
    "search_compounds": LabStockEnv._search,
    "get_compound": LabStockEnv._get_compound,
    "place_order": LabStockEnv._place_order,
    "request_approval": LabStockEnv._request_approval,
    "cancel_order": LabStockEnv._cancel_order,
    "get_budget": LabStockEnv._get_budget,
}


def _episode_from_task(task: Task, seed: int) -> _Episode:
    """Build the initial episode state from ``task.initial_state``.

    Initial order costs are computed from the compound prices. Initial stock is taken as
    already net of the initial placed orders.

    Raises:
        ValueError: If the initial state is malformed or its placed orders exceed the budget.
    """
    where = f"task {task.id!r} initial_state"
    state = task.initial_state
    unknown = set(state) - _INITIAL_STATE_KEYS
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    budget: Any = state.get("budget")
    if not _is_number(budget) or budget < 0:
        raise ValueError(f"{where}.budget must be a non-negative number")
    compounds = _read_compounds(state.get("compounds"), where)
    orders = _read_orders(state.get("orders", []), compounds, where)
    approvals = _read_approvals(state.get("approvals", []), compounds, where)
    spent = round(sum(o.cost for o in orders if o.status == "placed"), 2)
    if spent > budget + _MONEY_TOLERANCE:
        raise ValueError(f"{where}: placed orders cost {spent:.2f}, above budget {budget}")
    return _Episode(
        task_id=task.id,
        budget_total=float(budget),
        compounds=compounds,
        rng=random.Random(f"{seed}|{task.id}|labstock-flaky"),
        orders=orders,
        approvals=approvals,
    )


def _read_compounds(raw: Any, where: str) -> dict[str, _Compound]:
    """Validate the ``compounds`` list of an initial state."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}.compounds must be a non-empty list")
    compounds: dict[str, _Compound] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{where}.compounds entries must be mappings")
        missing = _COMPOUND_KEYS - set(item)
        if missing:
            raise ValueError(f"{where}.compounds entry missing keys {sorted(missing)}")
        cid = str(item["id"])
        if cid in compounds:
            raise ValueError(f"{where}: duplicate compound id {cid!r}")
        if item["hazard"] not in HAZARDS:
            raise ValueError(f"{where}: compound {cid!r} has unknown hazard {item['hazard']!r}")
        for key in ("stock_g", "price_per_g"):
            if not _is_number(item[key]) or item[key] < 0:
                raise ValueError(f"{where}: compound {cid!r} {key} must be a non-negative number")
        compounds[cid] = _Compound(
            id=cid,
            name=str(item["name"]),
            stock_g=float(item["stock_g"]),
            hazard=str(item["hazard"]),
            price_per_g=float(item["price_per_g"]),
        )
    return compounds


def _read_orders(raw: Any, compounds: dict[str, _Compound], where: str) -> list[_Order]:
    """Validate the initial ``orders`` list and compute each order's cost."""
    if not isinstance(raw, list):
        raise ValueError(f"{where}.orders must be a list")
    orders: list[_Order] = []
    for item in raw:
        if not isinstance(item, dict) or not {"order_id", "compound_id", "grams"} <= set(item):
            raise ValueError(f"{where}.orders entries need order_id, compound_id and grams")
        order_id = str(item["order_id"])
        if any(o.order_id == order_id for o in orders):
            raise ValueError(f"{where}: duplicate order id {order_id!r}")
        compound = compounds.get(str(item["compound_id"]))
        if compound is None:
            raise ValueError(f"{where}: order {order_id!r} refers to unknown compound")
        if not _is_number(item["grams"]) or item["grams"] <= 0:
            raise ValueError(f"{where}: order {order_id!r} grams must be positive")
        status = str(item.get("status", "placed"))
        if status not in ("placed", "cancelled"):
            raise ValueError(f"{where}: order {order_id!r} has unknown status {status!r}")
        grams = float(item["grams"])
        orders.append(
            _Order(
                order_id=order_id,
                compound_id=compound.id,
                grams=grams,
                cost=round(grams * compound.price_per_g, 2),
                status=status,
            )
        )
    return orders


def _read_approvals(raw: Any, compounds: dict[str, _Compound], where: str) -> list[_Approval]:
    """Validate the initial ``approvals`` list."""
    if not isinstance(raw, list):
        raise ValueError(f"{where}.approvals must be a list")
    approvals: list[_Approval] = []
    for item in raw:
        if not isinstance(item, dict) or not {"compound_id", "reason", "status"} <= set(item):
            raise ValueError(f"{where}.approvals entries need compound_id, reason and status")
        if str(item["compound_id"]) not in compounds:
            raise ValueError(f"{where}: approval refers to unknown compound")
        if item["status"] not in ("approved", "not_required"):
            raise ValueError(f"{where}: approval has unknown status {item['status']!r}")
        approvals.append(
            _Approval(str(item["compound_id"]), str(item["reason"]), str(item["status"]))
        )
    return approvals


def _check_arguments(parameters: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Check arguments against the small JSON-schema subset used by the tools.

    Returns:
        A description of the first problem found, or None when the arguments are valid.
    """
    properties: dict[str, dict[str, Any]] = parameters["properties"]
    for key in parameters["required"]:
        if key not in arguments:
            return f"missing {key!r}"
    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            return f"unexpected argument {key!r}"
        kind = spec["type"]
        if kind == "string" and not isinstance(value, str):
            return f"{key!r} must be a string"
        if kind == "number" and not _is_number(value):
            return f"{key!r} must be a number"
        if "minLength" in spec and len(value) < spec["minLength"]:
            return f"{key!r} is too short"
        if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
            return f"{key!r} must be greater than {spec['exclusiveMinimum']}"
    return None


def _has_approval(episode: _Episode, compound_id: str) -> bool:
    """Return True if an approved request exists for the compound."""
    return any(a.compound_id == compound_id and a.status == "approved" for a in episode.approvals)


def _next_order_id(episode: _Episode) -> str:
    """Return the next unused order id of the form ``O<n>``."""
    used = {o.order_id for o in episode.orders}
    n = len(episode.orders) + 1
    while f"O{n}" in used:
        n += 1
    return f"O{n}"


def _spent(episode: _Episode) -> float:
    """Return the total cost of placed orders, rounded to cents."""
    return round(sum(o.cost for o in episode.orders if o.status == "placed"), 2)


def _remaining(episode: _Episode) -> float:
    """Return the budget left after placed orders, rounded to cents."""
    return round(episode.budget_total - _spent(episode), 2)


def _budget_dict(episode: _Episode) -> dict[str, float]:
    """Return the budget as total, spent and remaining."""
    return {
        "total": round(episode.budget_total, 2),
        "spent": _spent(episode),
        "remaining": _remaining(episode),
    }


def _order_dict(order: _Order) -> dict[str, Any]:
    """Return an order as a plain dict."""
    return {
        "order_id": order.order_id,
        "compound_id": order.compound_id,
        "grams": order.grams,
        "cost": order.cost,
        "status": order.status,
    }


def _ok(call: ToolCall, content: Any) -> ToolResult:
    """Build a successful result."""
    return ToolResult(call_id=call.call_id, ok=True, content=content, error=None)


def _error(call: ToolCall, message: str) -> ToolResult:
    """Build a failed result with no content."""
    return ToolResult(call_id=call.call_id, ok=False, content=None, error=message)


def _canonical(arguments: dict[str, Any]) -> str:
    """Return a stable text form of an arguments object, used as a retry signature."""
    return repr(sorted(arguments.items()))


def _is_number(value: Any) -> bool:
    """Return True for finite ints and floats that are not bools."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
