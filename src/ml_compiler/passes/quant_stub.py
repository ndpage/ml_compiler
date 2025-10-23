from __future__ import annotations
from ml_compiler.ir.graph import Graph

# Simple stub: assign all float nodes a dummy scale and zero-point

def quant_stub(graph: Graph, scale: float = 0.01, zero_point: int = 0) -> Graph:
    for n in graph.nodes():
        if n.dtype == "float32" and n.op_type not in {"Const"}:  # skip constant weights for now
            n.quant = {"scale": scale, "zero_point": zero_point, "dtype": "int8"}
            n.dtype = "int8"  # pretend we've quantized
    return graph
