"""Convert a .tflite file to a C source + header with a byte array.

Usage:
  python tools/tflite_to_cc.py --in model.tflite --out model.cc --sym MODEL_TFLITE

This writes a .cc file containing `const unsigned char MODEL_TFLITE[] = { ... };`
and a header declaring the symbol and size.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def tflite_to_c_array(in_path: str, out_cc: str, out_h: str | None = None, sym_name: str = "MODEL_TFLITE") -> None:
    p = Path(in_path)
    data = p.read_bytes()
    arr = ", ".join(str(b) for b in data)
    # default to C++ style includes; caller may pass a .c extension for C output
    cc = f"// Generated from {p.name}\n#include <cstdint>\n#include <cstddef>\n\nconst unsigned char {sym_name}[] = {{ {arr} }};\nconst size_t {sym_name}_len = {len(data)};\n"
    Path(out_cc).write_text(cc, encoding="utf-8")
    if out_h:
        header = f"#pragma once\n#include <cstddef>\nextern const unsigned char {sym_name}[];\nextern const size_t {sym_name}_len;\n"
        Path(out_h).write_text(header, encoding="utf-8")
    print(f"Wrote {out_cc} and header {out_h}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .tflite file")
    ap.add_argument("--out_cc", required=True, help="Output C/C++ source file")
    ap.add_argument("--out_h", help="Output header file (optional)")
    ap.add_argument("--sym", default="MODEL_TFLITE", help="Symbol name for array")
    ap.add_argument("--format", choices=["c", "cpp"], default="cpp", help="Output language format: 'c' or 'cpp' (adjusts headers and guards)")
    args = ap.parse_args()
    # If user requests C output, produce C-style includes and header.
    if args.format == "c":
        # create C-compatible source (stdint.h, stddef.h) and header with include guards
        p = Path(args.input)
        data = p.read_bytes()
        arr = ", ".join(str(b) for b in data)
        cc = f"// Generated from {p.name}\n#include <stdint.h>\n#include <stddef.h>\n\nconst unsigned char {args.sym}[] = {{ {arr} }};\nconst size_t {args.sym}_len = {len(data)};\n"
        Path(args.out_cc).write_text(cc, encoding="utf-8")
        if args.out_h:
            guard = args.sym + "_H"
            header = (
                f"#ifndef {guard}\n#define {guard}\n\n#include <stddef.h>\n#ifdef __cplusplus\nextern \"C\" {{\n#endif\n\nextern const unsigned char {args.sym}[];\nextern const size_t {args.sym}_len;\n\n#ifdef __cplusplus\n}}\n#endif\n\n#endif // {guard}\n"
            )
            Path(args.out_h).write_text(header, encoding="utf-8")
        print(f"Wrote {args.out_cc} and header {args.out_h} (C format)")
    else:
        tflite_to_c_array(args.input, args.out_cc, args.out_h, args.sym)


if __name__ == "__main__":
    main()
