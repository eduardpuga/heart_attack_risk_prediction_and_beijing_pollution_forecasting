from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model from the file
with open('heart_attack/models/best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the fitted scaler
with open('heart_attack/models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def preprocess_data(data):
    """
    Preprocess the input data by scaling it using the same scaler 
    that was used during the model training.
    
    Args:
    data (DataFrame): The input data to scale.

    Returns:
    DataFrame: The scaled data.
    """
    data_scaled = scaler.transform(data)
    return data_scaled

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make predictions using the trained model.
    
    The input data is expected to be in JSON format, containing the features
    required by the model. The data is preprocessed and then passed to the
    model for prediction.

    Returns:
    JSON: The predicted class (or probability) as a JSON response.
    
    Example:
    POST /predict
    {
        "age": [63],
        "sex": [1],
        "cp": [3],
        "trtbps": [145],
        "chol": [233],
        "fbs": [1],
        "restecg": [0],
        "thalachh": [150],
        "exng": [0],
        "oldpeak": [2.3],
        "slp": [0],
        "caa": [0],
        "thall": [1]
    }
    """
    # Parse the JSON input data
    json_data = request.json
    
    # Convert the JSON data into a Pandas DataFrame
    data = pd.DataFrame(json_data)
    
    # Preprocess the data (e.g., scaling)
    data_preprocessed = preprocess_data(data)
    
    # Make predictions using the trained model
    prediction = model.predict(data_preprocessed)
    
    # Return the predictions as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    # Run the Flask app on the specified host and port
    app.run(host='0.0.0.0', port=5000)
