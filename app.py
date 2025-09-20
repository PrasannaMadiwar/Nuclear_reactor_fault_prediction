import pandas as pd
import numpy as np 
from flask import Flask
from flask  import  url_for, request, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl",'rb'))

features = ['Reactor Temp (°C)', 'Coolant Flow Rate (L/s)', 'Pressure (MPa)',
       'Radiation Level (μSv/h)', 'Turbine Speed (RPM)', 'Pump Status',
       'Power Output (MW)', 'Control Rod Position (%)',
       'Steam Flow Rate (kg/s)', 'Vibration Level (mm/s)', 'Water Level (m)']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    inputs = []
    missing = []

    for i in features:

        value = request.form.get(i)

        if value != None:
            inputs.append(value)
        else :
            missing.append(value)

    data = pd.DataFrame([inputs],columns=features)
    data_sc = scaler.transform(data)   
    pred = model.predict(data_sc)[0]
    pred_proba = model.predict_proba(data_sc)[0]

    if pred == 1 :
        prediction_label = 'ANAMOLY DETECTED'
        status = 'CRITICAL'
        status_class = 'critical'
        confidence = pred_proba[1] 

    else :
        prediction_label = 'NORMAL OPERATION'
        status = 'Normal'
        status_class = 'normal'
        confidence = pred_proba[0] 

    if confidence > 1:
        confidence = confidence / 100.0


    return render_template("predict.html", 
                         pred=prediction_label,
                         pred_proba=confidence,
                         inputs=inputs,
                         status=status,
                         status_class=status_class,
                         features=features)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port = int(os.environ.get('PORT',"5000")))