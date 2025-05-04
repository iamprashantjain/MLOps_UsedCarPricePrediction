from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

# Load preprocessor and model
preprocessor = joblib.load("artifacts/preprocessor/preprocessor.pkl")
model = joblib.load("artifacts/model/model.pkl")

@app.route('/')
def index():
    # Display the HTML form for input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form
        year = int(request.form['year'])
        odometer = float(request.form['odometer'])
        ownership = int(request.form['ownership'])  # 1 for Owned, 0 for Leased
        transmissionType = request.form['transmissionType']
        emiStartingValue = float(request.form['emiStartingValue'])
        emiEndingValue = float(request.form['emiEndingValue'])
        features = request.form.getlist('features')

        # Create a DataFrame with the input data (must match the structure of the training data)
        input_data = pd.DataFrame([{
            'year': year,
            'odometer': odometer,
            'ownership': ownership,
            'transmissionType': transmissionType,
            'emiStartingValue': emiStartingValue,
            'emiEndingValue': emiEndingValue,
            'features': features
        }])

        # Preprocess the input data
        transformed_data = preprocessor.transform(input_data)

        # Make the prediction using the trained model
        prediction = model.predict(transformed_data)

        # Return the prediction to the user
        return render_template('index.html', prediction_text=f"Predicted Listing Price: {prediction[0]:,.2f}")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)