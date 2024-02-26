import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('C:/Backup/Sepsis/Flask/sepsis.pkl', 'rb'))

@app.route('/')# route to display the home page
def home():
    return render_template('index.html') #rendering the home page

@app.route('/input')
def pred():
    return render_template('details.html')

@app.route('/predict',methods=["POST","GET"])# route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
   input_feature=[float(x) for x in request.form.values() ]  
   features_values=[np.array(input_feature)]
   print(features_values)
   names = [[ 'PRG','PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age',
    'Insurance']]
   data = pandas.DataFrame(features_values,columns=names) 

     # predictions using the loaded model file
   prediction=model.predict(data)
   prediction = prediction[0]
   if prediction == 0:
        return render_template('result.html', prediction_text = "After Analysing data we found that the Patient is Healthy.")
   else:
        return render_template('result.html', prediction_text = "The patient is suffering from Sepsis Disease.")
  
     # showing the prediction results in a UI
if __name__=="__main__":
        app.run(debug=False, port=5000)
    

 
