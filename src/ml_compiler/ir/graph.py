from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set


class GraphCycleError(Exception):
    pass


class UnsupportedOpError(Exception):
    pass


@dataclass
class Node:
    name: str
    op_type: str
    inputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    quant: Optional[Dict[str, Any]] = None  # placeholder for quantization info

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Node must have a name")


class Graph:
    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._inputs: Set[str] = set()
        self._outputs: Set[str] = set()

    # --- Node management ---
    def add_node(self, node: Node) -> None:
        if node.name in self._nodes:
            raise ValueError(f"Duplicate node name: {node.name}")
        self._nodes[node.name] = node

    def get_node(self, name: str) -> Node:
        return self._nodes[name]

    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    # --- Graph IO ---
    def mark_input(self, name: str) -> None:
        self._inputs.add(name)

    def mark_output(self, name: str) -> None:
        self._outputs.add(name)

    @property
    def inputs(self) -> List[str]:
        return list(self._inputs)

    @property
    def outputs(self) -> List[str]:
        return list(self._outputs)

    # --- Algorithms ---
    def topological_sort(self) -> List[Node]:
        # Kahn's algorithm
        indegree: Dict[str, int] = {n.name: 0 for n in self._nodes.values()}
        for n in self._nodes.values():
            for inp in n.inputs:
                if inp in indegree:
                    indegree[n.name] += 1
        ready = [name for name, deg in indegree.items() if deg == 0]
        order: List[Node] = []
        while ready:
            cur = ready.pop()
            order.append(self._nodes[cur])
            for other in self._nodes.values():
                if cur in other.inputs:
                    indegree[other.name] -= 1
                    if indegree[other.name] == 0:
                        ready.append(other.name)
        if len(order) != len(self._nodes):
            raise GraphCycleError("Graph has a cycle or disconnected components")
        return order

    def validate_acyclic(self) -> None:
        self.topological_sort()  # will raise if cyclic

    # --- Export (placeholder) ---
    def summary(self) -> str:
        lines = ["Graph Summary:"]
        lines.append(f"Inputs: {self.inputs}")
        lines.append(f"Outputs: {self.outputs}")
        for n in self.topological_sort():
            lines.append(f"{n.name}: {n.op_type} <- {n.inputs} shape={n.shape} dtype={n.dtype}")
        return "\n".join(lines)
