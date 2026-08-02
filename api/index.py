import io
import os

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, request
from PIL import Image

app = Flask(__name__)

CLASS_NAMES = ["Cat", "Dog"]
IMG_SIZE = (128, 128)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cats_dogs_model.tflite")

# Loaded once per cold start, not per-request.
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()


def predict(image: Image.Image):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(INPUT_DETAILS[0]["index"], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])

    prob_dog = float(prediction[0][0])
    predicted_class = CLASS_NAMES[int(prob_dog > 0.5)]
    confidence = prob_dog if predicted_class == "Dog" else 1 - prob_dog
    return predicted_class, confidence


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <title>Cats vs Dogs Classifier</title>
  <style>
    body {{ font-family: sans-serif; max-width: 480px; margin: 60px auto; text-align: center; }}
    img {{ max-width: 100%; border-radius: 8px; margin-top: 16px; }}
    .result {{ font-size: 1.3rem; margin-top: 16px; }}
    .error {{ color: #b00020; margin-top: 16px; }}
  </style>
</head>
<body>
  <h2>🐾 Cats vs Dogs Classifier</h2>
  <form action="/predict" method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required>
    <button type="submit">Predict</button>
  </form>
  {result_block}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return PAGE_TEMPLATE.format(result_block="")


@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("file")

    if file is None or file.filename == "":
        result_block = '<div class="error">No file uploaded — please choose an image.</div>'
        return PAGE_TEMPLATE.format(result_block=result_block)

    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        result_block = '<div class="error">Couldn\'t read that file as an image — try a JPG or PNG.</div>'
        return PAGE_TEMPLATE.format(result_block=result_block)

    predicted_class, confidence = predict(image)
    emoji = "🐱" if predicted_class == "Cat" else "🐶"

    result_block = f"""
    <div class="result">{emoji} Prediction: <b>{predicted_class}</b> ({confidence:.1%} confidence)</div>
    """
    return PAGE_TEMPLATE.format(result_block=result_block)


# Vercel's Python runtime looks for a WSGI-callable named "app" in this
# module — Flask's app object already satisfies that directly, no extra
# wrapper needed.
