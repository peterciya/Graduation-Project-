from flask import Flask, request, jsonify
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import joblib
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Chargez le modèle SARIMA

try:
    with open('sarima_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Modèle SARIMA chargé avec succès.")
except Exception as e:
    print("Erreur lors du chargement du modèle SARIMA :", e)

try:
    model1 = joblib.load('sarima_model_ages.pkl')
    print("Modèle SARIMA chargé avec succès.")
except Exception as e:
    print("Erreur lors du chargement du modèle SARIMA avec les sorties en zones :", e)

try:
    model2 = joblib.load('sarima_model_zones.pkl')
    print("Modèle SARIMA chargé avec succès.")
except Exception as e:
    print("Erreur lors du chargement du modèle SARIMA avec les sorties en ages :", e)

@app.route('/')
def home():
    return "Bienvenue sur l'API SARIMA!"



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Données reçues : %s", data)

        if not isinstance(data, dict) or 'steps' not in data or not isinstance(data['steps'], int):
            return jsonify({"error": "Le champ 'steps' est requis et doit être un entier."}), 400

        steps = data['steps']

        # Ensure 'model' is properly fitted SARIMAX model
        forecast = model.get_forecast(steps=steps)
        
        prediction = forecast.predicted_mean
        prediction_list = prediction.tolist()
        print("Prédiction : %s", prediction_list)
        return jsonify(prediction_list)

    except Exception as e:
        print("Erreur lors de la prédiction : %s", e)
        return jsonify({"error": str(e)}), 500
    

    
@app.route('/ages_predict', methods=['POST'])
def ages_predict():
    try:
        data = request.get_json()
        print("Données reçues : %s", data)

        if not isinstance(data, dict) or 'steps' not in data or not isinstance(data['steps'], int):
            return jsonify({"error": "Le champ 'steps' est requis et doit être un entier."}), 400

        steps = data['steps']

        # Ensure 'model' is properly fitted SARIMAX model
        forecast = model1.get_forecast(steps=steps)
        
        prediction = forecast.predicted_mean
        prediction_list = prediction.tolist()
        print("Prédiction : %s", prediction_list)
        return jsonify(prediction_list)

    except Exception as e:
        print("Erreur lors de la prédiction : %s", e)
        return jsonify({"error": str(e)}), 500
    

@app.route('/zones_predict', methods=['POST'])
def zones_predict():
    try:
        data = request.get_json()
        print("Données reçues : %s", data)

        if not isinstance(data, dict) or 'steps' not in data or not isinstance(data['steps'], int):
            return jsonify({"error": "Le champ 'steps' est requis et doit être un entier."}), 400

        steps = data['steps']

        # Ensure 'model' is properly fitted SARIMAX model
        forecast = model2.get_forecast(steps=steps)
        
        prediction = forecast.predicted_mean
        prediction_list = prediction.tolist()
        print("Prédiction : %s", prediction_list)
        return jsonify(prediction_list)

    except Exception as e:
        print("Erreur lors de la prédiction : %s", e)
        return jsonify({"error": str(e)}), 500
    
Flask(__name__)
if __name__ == '__main__':
    app.run(debug=True)
