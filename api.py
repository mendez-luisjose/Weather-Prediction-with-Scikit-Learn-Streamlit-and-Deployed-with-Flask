#API made with Flask

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from flask import Flask, request, jsonify
import pickle

MODEL_PATH = f'./model/weather_model.pkl'
SCALER_PATH = f'./model/scaler.pkl'

#Function to load the Model and the Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

#Prediction Function
def predict(input_array):
    input_array_scaled = scaler.transform(input_array)
    result = model.predict(input_array_scaled)
    prob_drizzle = round(model.predict_proba(input_array_scaled)[0][0], 2)
    prob_rain = round(model.predict_proba(input_array_scaled)[0][1], 2)
    prob_sun = round(model.predict_proba(input_array_scaled)[0][2], 2)
    prob_snow = round(model.predict_proba(input_array_scaled)[0][3], 2)
    prob_fog = round(model.predict_proba(input_array_scaled)[0][4], 2)

    print(result[0], prob_drizzle, prob_rain, prob_sun, prob_snow, prob_fog)
    
    results = {
        "result": int(result[0]),
        "prob_drizzle" : float(prob_drizzle),
        "prob_rain": float(prob_rain),
        "prob_sun": float(prob_sun),
        "prob_snow": float(prob_snow),
        "prob_fog": float(prob_fog)
    }

    return results

app = Flask(__name__)

#Flask App
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        np_array = np.array(data['array'])
        try:
            results = predict(np_array)
            print('Success!', 200)
            return jsonify({"Results": results})

    
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
