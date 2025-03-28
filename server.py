from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("attrition_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid format. Send JSON object."}), 400
    
    try:
        # Convert JSON data to numpy array
        features = np.array([list(data.values())]).astype(float)
        prediction = model.predict(features)
        return jsonify({"attrition_prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
