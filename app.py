import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model and scaler
with open("attrition_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to Employee Attrition Prediction API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get JSON input
    features = np.array(data["features"]).reshape(1, -1)

    # Preprocess input
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]
    return jsonify({"attrition": int(prediction)})  # Convert to int for JSON response

if __name__ == "__main__":
    app.run(debug=True)
