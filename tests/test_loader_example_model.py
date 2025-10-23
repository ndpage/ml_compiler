from pathlib import Path
from ml_compiler.frontend.onnx_loader import load_onnx_model


def test_example_cnn_loads():
    model_path = Path(__file__).parent.parent / "models" / "example_cnn.onnx"
    if not model_path.exists():
        # Generate the model inline (avoid importing heavy libs if already present)
        import numpy as np
        import onnx
        from onnx import helper, TensorProto
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])
        conv_w = np.random.randn(2, 1, 3, 3).astype(np.float32)
        conv_b = np.random.randn(2).astype(np.float32)
        conv_w_init = helper.make_tensor("conv_w", TensorProto.FLOAT, conv_w.shape, conv_w.flatten())
        conv_b_init = helper.make_tensor("conv_b", TensorProto.FLOAT, conv_b.shape, conv_b.flatten())
        fc_w = np.random.randn(72, 10).astype(np.float32)
        fc_b = np.random.randn(10).astype(np.float32)
        fc_w_init = helper.make_tensor("fc_w", TensorProto.FLOAT, fc_w.shape, fc_w.flatten())
        fc_b_init = helper.make_tensor("fc_b", TensorProto.FLOAT, fc_b.shape, fc_b.flatten())
        conv_node = helper.make_node("Conv", inputs=["input", "conv_w", "conv_b"], outputs=["conv_out"], pads=[0,0,0,0], strides=[1,1])
        relu_node = helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"]) 
        flatten_node = helper.make_node("Flatten", inputs=["relu_out"], outputs=["flat_out"], axis=1)
        gemm_node = helper.make_node("Gemm", inputs=["flat_out", "fc_w", "fc_b"], outputs=["logits"], alpha=1.0, beta=1.0)
        softmax_node = helper.make_node("Softmax", inputs=["logits"], outputs=["probs"], axis=1)
        output_tensor = helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 10])
        graph_def = helper.make_graph([conv_node, relu_node, flatten_node, gemm_node, softmax_node], "TinyCNN", [input_tensor], [output_tensor], [conv_w_init, conv_b_init, fc_w_init, fc_b_init])
        model = helper.make_model(graph_def, producer_name="ml_compiler_example_test")
        onnx.save(model, str(model_path))
    graph = load_onnx_model(str(model_path))
    op_types = [n.op_type for n in graph.nodes()]
    # Ensure all expected ops present (Const initializers + sequence ops)
    assert "Conv" in op_types
    assert "Relu" in op_types
    assert "Flatten" in op_types
    assert "Gemm" in op_types
    assert "Softmax" in op_types
