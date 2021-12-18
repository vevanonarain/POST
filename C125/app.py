from flask import Flask, jsonify, request
from classifier import getPrediction

app = Flask(__name__)
@app.route("/predict", methods = ["POST"])

def predictDigit():
    image = request.files.get("Alphabet")
    pred = getPrediction(image)
    return jsonify({
        "prediction": pred
    })

app.run()