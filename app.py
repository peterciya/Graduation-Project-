from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

models_ages = {}
model_directory = 'ages'
if not os.path.exists(model_directory):
    print(f"Le dossier {model_directory} n'existe pas.")
else:
    for filename in os.listdir(model_directory):
        if filename.endswith('.pkl'):
            try:
                ages = os.path.splitext(filename)[0]
                model_path = os.path.join(model_directory, filename)
                print(f"Chargement du modèle : {model_path}")
                models_ages[ages] = joblib.load(model_path)
                print(f"Modèle pour l'âge {ages} chargé avec succès.")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle {ages}: {e}")

# Charger les modèles depuis le dossier 'ages'
#models_ages = {}
#for filename in os.listdir('ages'):
  #  if filename.endswith('.pkl'):
   #     ages = os.path.splitext(filename)[0]
    #    models_ages[ages] = joblib.load(os.path.join('ages', filename))


@app.route('/')
def home():
    return "Bienvenue sur l'API SARIMA!"


@app.route('/predict_ages', methods=['POST'])
def predict_ages():
    data = request.json
    periods = data['periods']
    
    results = {}
    
    # Faire des prévisions pour tous les modèles
    for column, model_age in models_ages.items():
        if model_age is not None:
            forecast = model_age.get_forecast(steps=periods)
            forecast_values = forecast.predicted_mean
            forecast_values = forecast_values.apply(lambda x: np.floor(x) if (x - np.floor(x)) < 0.5 else np.ceil(x))
            forecast_dates = pd.date_range(start='2024-07-31', periods=periods, freq='M')
            forecast_series = pd.Series(forecast_values.values, index=forecast_dates)
            forecast_series.index = forecast_series.index.strftime('%Y-%m-%d')  
            results[column] = forecast_series.to_dict()
        else:
            results[column] = "Modèle non trouvé ou non valide"
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
desfqw2