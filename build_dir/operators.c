#include "operators.h"
void relu(float* out, const float* inp, int size) { for(int i=0;i<size;++i){ out[i] = inp[i] > 0 ? inp[i] : 0; } }

