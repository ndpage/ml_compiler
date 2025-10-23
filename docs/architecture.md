# Architecture Overview

## Layers
1. Frontend
   - Loader for ONNX models (`frontend.onnx_loader`) producing internal IR.
2. Intermediate Representation (IR)
   - Simple graph of `Node` objects inside a `Graph`.
   - Provides utilities: add_node, get_node, topological_sort, export.
3. Passes
   - Operate on IR in-place or return transformed graph.
   - Initial passes: quant_stub, fuse_conv_relu (placeholder), shape_infer (basic), memory_plan (future).
4. Code Generation
   - Translates IR into C code for ESP32.
   - Emission pieces:
     * `operators.c/.h` with per-op kernels (naive reference)
     * `model.c` with static weight arrays & inference function
     * `main.c` example
5. CLI
   - `mlc compile --model path.onnx --out build/ --passes quant_stub,fuse_conv_relu`.

## Data Flow
ONNX -> Frontend Parser -> IR Graph -> Pass Pipeline -> Codegen -> C Artifacts -> User builds with ESP-IDF / gcc.

## IR Node Fields (draft)
```
Node(
  name: str,
  op_type: str,
  inputs: List[str],
  attrs: Dict[str, Any],
  shape: Tuple[int, ...],
  dtype: str = 'float32',
  quant: Optional[QuantInfo] = None
)
```
QuantInfo will later hold scale/zero_point/per-channel arrays.

## Pass Pipeline Concept
Represent passes as callables: `graph = pass_fn(graph, **kwargs)`; maintain ordering; allow CLI selection.

## Error Handling Strategy
- Validate supported ops at load time; raise `UnsupportedOpError`.
- Ensure acyclic graph via DFS in topological_sort (raise `GraphCycleError`).

## Future Expansion
- Multi-target backend registry (ESP32, STM32, RP2040) mapping ops to platform-specific kernels.
- Memory planner adds buffer reuse metadata -> codegen uses arena.
- Profiling: instrumentation macros around kernels.

## Testing Strategy
- Unit tests for IR operations and loader mapping.
- Golden output tests for codegen (compare generated C skeleton).

## Performance Considerations (Later)
- Convolution lowering to im2col + Gemm for simplicity, then optimized direct kernels.
- Optional INT8 path after quantization becomes real.
