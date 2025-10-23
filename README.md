# ml_compiler

Experimental ML compiler to turn ONNX models into C code targeting ESP32 and other microcontrollers (future). Inspired by TVM / Glow concepts but drastically simplified for educational / exploratory purposes.

## Why
Running small neural nets on constrained devices requires:
- Reduced memory footprint
- Integer quantization
- Elimination of heavyweight runtimes

This project explores a minimal pipeline from ONNX to portable C suitable for integration with ESP-IDF or other embedded SDKs.

## Status
MVP skeleton:
- IR graph (`Node`, `Graph`)
- ONNX loader (subset ops) -> internal IR
- Quantization stub pass (`quant_stub`)
- Basic C code generator (`generate_c`) producing operator stubs & inference loop
- CLI: `python -m ml_compiler.cli compile --model model.onnx --out build --passes quant_stub`

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[onxx,dev]
```

(Note: optional extra dependencies spelled `onxx` due to placeholder; adjust to `onnx` if editing `pyproject.toml`.)

## Usage
```bash
python -m ml_compiler.cli compile --model path/to/model.onnx --out build_dir --passes quant_stub
```
Outputs C files in `build_dir`.

## Directory Structure
```
src/ml_compiler/
  ir/           # Graph & Node
  frontend/     # ONNX loader
  passes/       # Transform passes (quant, fusion TBD)
  codegen/      # ESP32 C emission
  cli.py        # Command line interface
```

## Roadmap
1. Shape inference pass
2. Operator fusion (Conv + Relu)
3. Memory planner (buffer liveness, arena allocation)
4. Real quantization (calibration tooling, int8 kernels)
5. Expand operator coverage (Pooling, Softmax refine, BatchNorm folding)
6. Backend abstraction & optimized kernels (ESP-DSP, CMSIS-NN, custom intrinsics)
7. Profiling & benchmarking harness (cycle counts, memory usage)
8. Support alternative targets (STM32, RP2040, nRF52)
9. Tensor layout optimization (NCHW vs NHWC selection)
10. Compress weights (delta encoding / sparsity exploration)

## Contributing
PRs welcome. Keep changes modular; add tests for new passes or codegen behaviors. Avoid premature optimization; focus on correctness then iterate.

## License
MIT
