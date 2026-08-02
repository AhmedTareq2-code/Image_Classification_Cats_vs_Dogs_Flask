import numpy as np
from PIL import Image

try:
    # ai-edge-litert — the maintained, lightweight interpreter used in
    # requirements.txt for this project (successor to tflite-runtime),
    # good fit for serverless deployments like Vercel
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    # Falls back to the interpreter bundled with full TensorFlow if
    # ai-edge-litert isn't installed (e.g. local dev)
    from tensorflow.lite.python.interpreter import Interpreter

CLASS_NAMES = ["Cat", "Dog"]        # alphabetical folder order — must match training
IMG_SIZE = (128, 128)               # must match the size used during training
MODEL_PATH = "cats_dogs_model.tflite"

# ---------------------------------------------------------------------------
# Load the TFLite model once, at import time
# ---------------------------------------------------------------------------
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict(image: Image.Image):
    """Same preprocessing as the original predict_image() function —
    resize to 128x128, rescale to [0,1], add a batch dimension."""
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])

    prob_dog = float(prediction[0][0])
    predicted_class = CLASS_NAMES[int(prob_dog > 0.5)]
    confidence = prob_dog if predicted_class == "Dog" else 1 - prob_dog

    return predicted_class, confidence
