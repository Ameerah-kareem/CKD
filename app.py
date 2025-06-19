from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('ckd_model.pkl')

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Expecting JSON data with patient information
    data = request.get_json(force=True)
    
    # Create an input vector in the same order as the training features
    # You may need to validate and convert the input accordingly
    try:
        input_vector = [
            float(data.get('age', 0)),
            float(data.get('bp', 0)),
            float(data.get('sg', 0)),
            float(data.get('al', 0)),
            float(data.get('su', 0)),
            float(data.get('bgr', 0)),
            float(data.get('bu', 0)),
            float(data.get('sc', 0)),
            float(data.get('sod', 0)),
            float(data.get('pot', 0)),
            float(data.get('hemo', 0)),
            float(data.get('pcv', 0)),
            float(data.get('wc', 0)),
            float(data.get('rc', 0)),
            float(data.get('rbc', 0)),
            float(data.get('pc', 0)),
            float(data.get('pcc', 0)),
            float(data.get('ba', 0)),
            float(data.get('htn', 0)),
            float(data.get('dm', 0)),
            float(data.get('cad', 0)),
            float(data.get('appet', 0)),
            float(data.get('pe', 0)),
            float(data.get('ane', 0))
        ]
    except Exception as e:
        return jsonify({"error": f"Invalid input format: {e}"}), 400
    
    # Reshape the input for prediction (1 sample with many features)
    input_vector = np.array(input_vector).reshape(1, -1)
    prediction = model.predict(input_vector)[0]
    
    # Return the prediction result
    result = "CKD Detected" if prediction == 1 else "No CKD Detected"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
