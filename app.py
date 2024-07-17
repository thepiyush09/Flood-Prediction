from flask import Flask, request, render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load("flood_prediction_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        city = request.form['city']
        features = [
            request.form['MonsoonIntensity'],
            request.form['TopographyDrainage'],
            request.form['RiverManagement'],
            request.form['Deforestation'],
            request.form['Urbanization'],
            request.form['ClimateChange'],
            request.form['DamsQuality'],
            request.form['Siltation'],
            request.form['AgriculturalPractices'],
            request.form['Encroachments'],
            request.form['IneffectiveDisasterPreparedness'],
            request.form['DrainageSystems'],
            request.form['CoastalVulnerability'],
            request.form['Landslides'],
            request.form['Watersheds'],
            request.form['DeterioratingInfrastructure'],
            request.form['PopulationScore'],
            request.form['WetlandLoss'],
            request.form['InadequatePlanning'],
            request.form['PoliticalFactors']
        ]
        
        # Convert features to appropriate data types
        features = list(map(float, features))
        
        # Convert to numpy array and reshape for the model
        features_array = np.array(features).reshape(1, -1)
        
        # Predict using the model
        prediction = model.predict(features_array)[0]
        
        return render_template('result.html', city=city, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
