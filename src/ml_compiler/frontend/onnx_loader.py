from __future__ import annotations
from typing import List

try:
    import onnx  # type: ignore
except ImportError:  # pragma: no cover
    onnx = None  # type: ignore

from ml_compiler.ir.graph import Graph, Node, UnsupportedOpError

SUPPORTED_OPS = {"Conv", "Relu", "Gemm", "MatMul", "Add", "Flatten", "MaxPool", "Softmax"}


def load_onnx_model(path: str) -> Graph:
    if onnx is None:
        raise RuntimeError("onnx package not installed. Install with: pip install onnx")
    model = onnx.load(path)
    graph = Graph()

    # Record inputs
    for inp in model.graph.input:
        name = inp.name
        graph.mark_input(name)

    # Process initializers (weights) as constant nodes
    for init in model.graph.initializer:
        # For simplicity, treat weights as nodes with no inputs
        shape = tuple(dim for dim in init.dims)
        n = Node(name=init.name, op_type="Const", inputs=[], attrs={"data_type": init.data_type}, shape=shape)
        graph.add_node(n)

    # Iterate nodes
    for node in model.graph.node:
        op_type = node.op_type
        if op_type not in SUPPORTED_OPS:
            raise UnsupportedOpError(f"Unsupported op: {op_type}")
        inputs: List[str] = list(node.input)
        outputs: List[str] = list(node.output)
        attrs = {}
        for attr in node.attribute:
            # Basic attr extraction
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
        # We assume single output for simplicity
        out_name = outputs[0]
        gnode = Node(name=out_name, op_type=op_type, inputs=inputs, attrs=attrs)
        graph.add_node(gnode)

    # Mark graph outputs
    for out in model.graph.output:
        graph.mark_output(out.name)

    return graph
