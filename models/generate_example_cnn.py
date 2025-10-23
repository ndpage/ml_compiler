"""Generate a tiny CNN ONNX model: Conv -> Relu -> Flatten -> Gemm -> Softmax.
Uses random weights and fixed input shape (1,1,8,8).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto

# Input tensor
input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])

# Weights for conv: out_channels=2, in_channels=1, kernel=3x3
conv_w = np.random.randn(2, 1, 3, 3).astype(np.float32)
conv_b = np.random.randn(2).astype(np.float32)

conv_w_init = helper.make_tensor("conv_w", TensorProto.FLOAT, conv_w.shape, conv_w.flatten())
conv_b_init = helper.make_tensor("conv_b", TensorProto.FLOAT, conv_b.shape, conv_b.flatten())

# After conv (stride=1, pad=0): output shape (1,2,6,6)
# Flatten -> (1, 72)
# Gemm to 10 classes
fc_w = np.random.randn(72, 10).astype(np.float32)
fc_b = np.random.randn(10).astype(np.float32)

fc_w_init = helper.make_tensor("fc_w", TensorProto.FLOAT, fc_w.shape, fc_w.flatten())
fc_b_init = helper.make_tensor("fc_b", TensorProto.FLOAT, fc_b.shape, fc_b.flatten())

conv_node = helper.make_node(
    "Conv",
    inputs=["input", "conv_w", "conv_b"],
    outputs=["conv_out"],
    pads=[0, 0, 0, 0],
    strides=[1, 1],
)
relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"]) 
flatten_node = helper.make_node("Flatten", inputs=["relu_out"], outputs=["flat_out"], axis=1)
# Gemm: Y = alpha*A*B + beta*C ; We'll treat inputs as (flat_out, fc_w, fc_b)
# ONNX Gemm expects A,B,C shapes: (M,K),(K,N),(N) -> output (M,N)
# A=(1,72), B=(72,10), C=(10)
gemm_node = helper.make_node("Gemm", inputs=["flat_out", "fc_w", "fc_b"], outputs=["logits"], alpha=1.0, beta=1.0)
softmax_node = helper.make_node("Softmax", inputs=["logits"], outputs=["probs"], axis=1)

output_tensor = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 10])

graph = helper.make_graph(
    [conv_node, relu_node, flatten_node, gemm_node, softmax_node],
    "TinyCNN",
    [input_tensor],
    [output_tensor],
    [conv_w_init, conv_b_init, fc_w_init, fc_b_init],
)

model = helper.make_model(graph, producer_name="ml_compiler_example")

def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate tiny CNN ONNX model")
    parser.add_argument("--out", default="models/example_cnn.onnx", help="Output ONNX file path")
    args = parser.parse_args()
    out_path = Path(args.out)
    # Ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"Generated {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
