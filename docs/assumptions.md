# Initial Scope & Assumptions

## Goal
Build an ML compiler that takes a high-level neural network representation (initially ONNX) and produces optimized C code runnable on ESP32-class microcontrollers, with a roadmap to support other embedded / IoT targets.

## Why not direct TensorRT?
TensorRT is NVIDIA GPU focused (CUDA kernels, GPU memory mgmt). ESP32 is MCU (Xtensa / RISC-V cores, limited RAM, no CUDA). Therefore: we cannot *execute* TensorRT engines on ESP32. Instead we:
- Accept ONNX (exported from PyTorch or TensorFlow) as the front-end.
- Perform our own lightweight optimization passes.
- Generate portable C (later: optional CMSIS-NN, ESP-DSP, custom kernels).

Long-term we could have a "compat layer" that accepts a subset of TensorRT engine description if useful, but not now.

## Scope (MVP)
- Support small feedforward / CNN models (Conv, Relu, Gemm/FC, Add, MaxPool, Flatten, Softmax subset).
- Static shapes (no dynamic batch dims).
- Inference only.
- Quantization: start with fake/stub annotation -> later implement real int8 symmetric quant.
- Memory planning: naive (malloc / static arrays) initially.
- Codegen: single-threaded C with simple operator function stubs.

## Non-Goals (initially)
- Training / backprop.
- Runtime operator scheduling beyond topological order.
- Advanced graph transformations (only simple fusion prototypes).
- Hardware-specific intrinsics (later optimization phase).

## IR Design (initial sketch)
Graph of Nodes, each Node has:
- name
- op_type
- inputs (list of node names)
- attrs (dict)
- tensor info (shape, dtype)
- optional quantization params

## Risks & Considerations
- Limited RAM on ESP32 -> need future memory reuse & arena allocator.
- Quantization accuracy trade-offs -> need calibration tools later.
- Operator coverage creep -> keep tight initial set.

## Roadmap (High Level)
1. ONNX -> Internal IR loader
2. Passes: shape inference (basic), quantization stub, operator fusion (Conv+Relu)
3. Codegen: C skeleton (operators + main inference loop)
4. Memory planner (liveness analysis -> buffer reuse)
5. Real quantization (scale/zero-point, per-channel for Conv)
6. Add alternative backends (ESP-DSP, CMSIS-NN mapping)
7. Extend to other targets (STM32, Nordic nRF, RP2040)
8. Benchmark harness & profiling hooks
