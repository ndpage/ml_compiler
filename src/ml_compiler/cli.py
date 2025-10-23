from __future__ import annotations
import argparse
from ml_compiler.frontend.onnx_loader import load_onnx_model
from ml_compiler.passes.quant_stub import quant_stub
from ml_compiler.codegen.esp32_c import generate_c

PASS_REGISTRY = {
    "quant_stub": quant_stub,
}


def run_passes(graph, passes: list[str]):
    for p in passes:
        fn = PASS_REGISTRY.get(p)
        if fn is None:
            raise ValueError(f"Unknown pass: {p}")
        graph = fn(graph)
    return graph


def main():  # pragma: no cover (CLI wrapper)
    ap = argparse.ArgumentParser(description="ml_compiler CLI")
    sub = ap.add_subparsers(dest="command")

    comp = sub.add_parser("compile", help="Compile ONNX model to C")
    comp.add_argument("--model", required=True, help="Path to ONNX model")
    comp.add_argument("--out", required=True, help="Output directory for generated C")
    comp.add_argument("--passes", default="", help="Comma separated passes, e.g. quant_stub")

    args = ap.parse_args()
    if args.command == "compile":
        graph = load_onnx_model(args.model)
        passes = [p for p in args.passes.split(",") if p]
        graph = run_passes(graph, passes)
        generate_c(graph, args.out)
        print("Compilation complete.")
    else:
        ap.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
