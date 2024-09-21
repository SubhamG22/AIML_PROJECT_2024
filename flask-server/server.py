from flask import Flask, request, jsonify, send_from_directory,  render_template
from flask_cors import CORS  # Import CORS for cross-origin resource sharing
import joblib
import pandas as pd 
import os

app = Flask(__name__, static_folder='../client/build/',  template_folder='../client/build')
CORS(app)  # Apply CORS to the entire app

# Load models and encoder
crop_model = joblib.load('crop_model.pkl')
fertilizer_model = joblib.load('fertilizer_model.pkl')
ingredient_model = joblib.load('ingredient_model.pkl')
soil_type_encoder = joblib.load('soil_type_encoder.pkl')

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_react_app(path):
    return render_template('index.html')


# Serve static files for React (CSS, JS, etc.)
@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(os.path.join(app.static_folder, 'static'), path)
  
# Define routes
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.json
    input_data = pd.DataFrame([data])
    input_data['soil_type'] = soil_type_encoder.transform(input_data['soil_type'])
    prediction = crop_model.predict(input_data)
    return jsonify({'crop': prediction[0]})

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    data = request.json
    input_data = pd.DataFrame([data])
    input_data['soil_type'] = soil_type_encoder.transform(input_data['soil_type'])
    prediction = fertilizer_model.predict(input_data)
    return jsonify({'fertilizer': prediction[0]})

@app.route('/predict_ingredient', methods=['POST'])
def predict_ingredient():
    # Ensure 'data' is correctly defined and assigned
    data = request.json  # Retrieve the JSON data from the request
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Convert JSON data to DataFrame
        input_data = pd.DataFrame([data])
        input_data['soil_type'] = soil_type_encoder.transform(input_data['soil_type'])
        
        # Predict and return the result
        prediction = ingredient_model.predict(input_data)
        return jsonify({'ingredient': prediction[0]})
    except Exception as e:
        # Handle exceptions and provide feedback
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
