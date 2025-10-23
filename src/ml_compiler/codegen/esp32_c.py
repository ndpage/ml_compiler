from __future__ import annotations
from pathlib import Path
from ml_compiler.ir.graph import Graph

# Very naive code generator creating a few C source files.

OP_KERNEL_TEMPLATES = {
    "Relu": "void relu(float* out, const float* inp, int size) { for(int i=0;i<size;++i){ out[i] = inp[i] > 0 ? inp[i] : 0; } }",
    # Placeholders for other ops
}

HEADER_GUARD = "#pragma once\n#include <stdint.h>\n#include <stddef.h>\n"


def generate_c(graph: Graph, out_dir: str) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    # operators.c / operators.h
    op_h = path / "operators.h"
    op_c = path / "operators.c"

    with op_h.open("w") as fh:
        fh.write(HEADER_GUARD)
        fh.write("// Operator declarations\n")
        fh.write("void relu(float* out, const float* inp, int size);\n")

    with op_c.open("w") as fc:
        fc.write("#include \"operators.h\"\n")
        for code in OP_KERNEL_TEMPLATES.values():
            fc.write(code + "\n\n")

    # model.c (very simplified loop over nodes)
    model_c = path / "model.c"
    with model_c.open("w") as mc:
        mc.write("#include <stdio.h>\n#include \"operators.h\"\n\n")
        mc.write("// Dummy buffers (real code would size per node)\n")
        mc.write("float input_buffer[1024];\nfloat output_buffer[1024];\n\n")
        mc.write("void run_inference() {\n")
        for node in graph.topological_sort():
            if node.op_type == "Relu":
                mc.write("    // Relu op\n    relu(output_buffer, input_buffer, 1024);\n")
            else:
                mc.write(f"    // {node.op_type} not implemented\n")
        mc.write("}\n")

    # main.c example
    main_c = path / "main.c"
    with main_c.open("w") as mainf:
        mainf.write("#include <stdio.h>\n\nvoid run_inference();\n\nint main(){\n    run_inference();\n    printf(\"Inference complete\\n\");\n    return 0;\n}\n")

