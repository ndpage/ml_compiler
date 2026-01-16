#include <stdio.h>
#include "operators.h"

// Dummy buffers (real code would size per node)
float input_buffer[1024];
float output_buffer[1024];

void run_inference() {
    // Const not implemented
    // Const not implemented
    // Const not implemented
    // Const not implemented
    // Conv not implemented
    // Relu op
    relu(output_buffer, input_buffer, 1024);
    // Flatten not implemented
    // Gemm not implemented
    // Softmax not implemented
}
