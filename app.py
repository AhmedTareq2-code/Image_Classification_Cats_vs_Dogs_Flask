from flask import Flask, request, render_template, jsonify
from PIL import Image

from model import predict

app = Flask(__name__)


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
