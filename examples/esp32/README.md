ESP32 integration notes (TFLite Micro)

Overview
- Use TensorFlow Lite Micro (TFLM) library compiled into your ESP-IDF project as a component.
- Convert your model to a `.tflite` file, optionally full integer quantized.
- Convert the `.tflite` to a C array using `tools/tflite_to_cc.py` and include it in your component.

Steps
1. Convert model to .tflite using `tools/convert_to_tflite.py`.
2. Generate C array:
   python tools/tflite_to_cc.py --input model.tflite --out_cc model.cc --out_h model.h --sym MY_MODEL_TFLITE
3. Add TFLite Micro as a component in ESP-IDF or vendor port. The easiest route is to use the Arduino_TensorFlowLite or the official TFLM port for ESP32.
4. Include `model.cc` in your component sources and include `model.h` where you instantiate the interpreter.

Minimal example snippet (C++):

#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

const tflite::Model* model = tflite::GetModel(MY_MODEL_TFLITE);
static tflite::MicroMutableOpResolver<10> resolver; // adjust size for ops used
static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);
interpreter.AllocateTensors();

// Copy input into interpreter->input(0)->data.f and invoke
interpreter.Invoke();

Notes
- Adjust resolver to include only the ops your model needs. This reduces binary size.
- For quantized models, use int8 input buffer and proper dequantization.
- If you want LiteRT instead of TFLM, the flow for mobile (iOS/Android) will be different; see mobile docs.
