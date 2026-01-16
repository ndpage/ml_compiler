Mobile integration notes (iOS / Android) for TFLite / LiteRT

iOS (Swift / Obj-C)
- Use TensorFlow Lite CocoaPod or build the static framework from source.
- Add the `.tflite` model to your app bundle.
- Use the Swift Task Library or the C API to load the model and run inference.

Android (Java/Kotlin)
- Add dependency in Gradle: `implementation 'org.tensorflow:tensorflow-lite:2.x.x'` and optionally `tensorflow-lite-gpu` or `tensorflow-lite-task-vision`.
- Place the `.tflite` model in `src/main/assets` and load with `Interpreter`.

LiteRT / Task Library
- The newer Task Library simplifies model usage for mobile (pre/post processing). Check the official TF docs for the Task API usage.

Cross-platform tips
- Ensure you use the same ops and quantization when running on microcontrollers and mobile; prefer int8 quantized model for parity where possible.
- For mobile, you can use NNAPI / GPU delegates; for MCUs you rely on TFLM runtime.

References
- TensorFlow Lite guides: https://www.tensorflow.org/lite
- Task library docs: https://www.tensorflow.org/lite/inference_with_delegate
