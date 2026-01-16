"""Representative dataset example for quantization.

Provide a `representative_dataset()` generator that yields input samples compatible
with the model's input signature. Modify this to load real samples for accurate
quantization.
"""
from __future__ import annotations
import numpy as np
from typing import Iterator


def representative_data(num_samples: int = 100) -> Iterator[tuple]:
    """Yield tuples/lists matching model input signature.

    Default yields random data shaped (1,1,8,8) with values in [0,1).
    Replace with actual preprocessing of your dataset for accurate quantization.
    """
    for _ in range(num_samples):
        sample = np.random.rand(1, 1, 8, 8).astype(np.float32)
        # TFLite representative_dataset should yield a list or tuple of inputs
        yield [sample]


# Convenience wrapper expected by convert_to_tflite script
def representative_dataset():
    return lambda: (x for x in representative_data())
