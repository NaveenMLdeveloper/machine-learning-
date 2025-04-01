from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model_path = "best_model.pkl"
scaler_path = "scaler.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_data = scaler.transform(np.array(data).reshape(1, -1))
    prediction = model.predict(scaled_data)
    output = 'Presence of breast cancer (malignant tumor).' if prediction[0] == 1 else 'Absence of breast cancer (benign tumor).'
    return render_template('index.html', prediction_text=f'Breast Cancer Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
