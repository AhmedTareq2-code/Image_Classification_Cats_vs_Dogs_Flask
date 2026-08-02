import io
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify

try:
    # Lightweight interpreter — preferred for serverless deployments (e.g. Vercel)
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Falls back to the interpreter bundled with full TensorFlow if
    # tflite-runtime isn't installed (e.g. local dev)
    from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

CLASS_NAMES = ["Cat", "Dog"]        # alphabetical folder order — must match training
IMG_SIZE = (128, 128)               # must match the size used during training
MODEL_PATH = "cats_dogs_model.tflite"

# ---------------------------------------------------------------------------
# Load the TFLite model once at startup
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        image = Image.open(file.stream)
        predicted_class, confidence = predict(image)
    except Exception as exc:
        return jsonify({"error": f"Could not process image: {exc}"}), 400

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "low_confidence": confidence < 0.6,
    })


if __name__ == "__main__":
    app.run(debug=True)
