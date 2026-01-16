"""Convert a SavedModel or ONNX model to a .tflite file, with optional full-integer quantization.

Usage examples:
  # Convert SavedModel to float TFLite
  python tools/convert_to_tflite.py --saved_model path/to/saved_model --out model.tflite

  # Convert ONNX -> SavedModel (requires onnx-tf), then to float TFLite
  python tools/convert_to_tflite.py --onnx model.onnx --out model.tflite

  # Convert and apply full integer quantization using a representative dataset generator
  python tools/convert_to_tflite.py --saved_model path/to/saved_model --out model_int8.tflite --quant --rep_module tools.representative_data

Notes:
- For ONNX input the script will try to use onnx-tf to convert to a TF SavedModel.
- Representative dataset module should provide a callable `representative_dataset()` that yields a sequence of input tuples or ndarrays.
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path


def convert_saved_model_to_tflite(saved_model_dir: str, out_path: str, quant: bool = False, rep_module: str | None = None):
    try:
        import tensorflow as tf
    except Exception as e:  # pragma: no cover - environment-dependent
        print("TensorFlow not available. Install with: pip install tensorflow")
        raise

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quant:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Use full integer quantization if a representative dataset is provided
        if rep_module is None:
            raise ValueError("Full integer quantization requires --rep_module specifying a module that provides `representative_dataset()`")
        # import representative dataset generator
        spec = importlib.import_module(rep_module)
        if not hasattr(spec, "representative_dataset"):
            raise ValueError("Representative module must expose a `representative_dataset()` function")
        rep_gen = spec.representative_dataset
        converter.representative_dataset = rep_gen
        # Target int8
        try:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        except Exception:
            # Older TF versions may vary
            pass

    tflite_model = converter.convert()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {out_path}")


def convert_onnx_to_saved_model(onnx_path: str, saved_model_out: str):
    # Try to convert ONNX to TF SavedModel using onnx-tf
    try:
        import onnx
    except Exception:
        raise RuntimeError("onnx package required for ONNX conversion. Install: pip install onnx")

    try:
        # onnx-tf can be used as a converter
        from onnx_tf.backend import prepare
    except Exception:
        raise RuntimeError("onnx-tf required for ONNX->SavedModel conversion. Install: pip install onnx-tf")

    model = onnx.load(onnx_path)
    tf_rep = prepare(model)  # this is the onnx-tf representation
    # tf_rep.export_graph saves as SavedModel
    tf_rep.export_graph(saved_model_out)
    print(f"Converted ONNX -> SavedModel at {saved_model_out}")


def main():  # pragma: no cover
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--saved_model", help="Path to TensorFlow SavedModel directory")
    group.add_argument("--onnx", help="Path to ONNX model file to convert")
    ap.add_argument("--out", required=True, help="Output .tflite path")
    ap.add_argument("--quant", action="store_true", help="Enable full integer quantization (requires --rep_module)")
    ap.add_argument("--rep_module", help="Python module path that exposes representative_dataset() (e.g. tools.representative_data)")
    ap.add_argument("--tmp_saved_model", default="/tmp/onnx_saved_model", help="Temporary SavedModel output for ONNX conversion")
    args = ap.parse_args()

    if args.onnx:
        convert_onnx_to_saved_model(args.onnx, args.tmp_saved_model)
        saved_model_dir = args.tmp_saved_model
    else:
        saved_model_dir = args.saved_model

    convert_saved_model_to_tflite(saved_model_dir, args.out, quant=args.quant, rep_module=args.rep_module)


if __name__ == "__main__":  # pragma: no cover
    main()
