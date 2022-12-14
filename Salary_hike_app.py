# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:19:37 2022

@author: Hi
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI

    Returns
    -------
    None.

    '''
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output= np.round(prediction[0],2)
    
    
    return render_template('index.html', prediction_text='Employee Salary Should be $ {}'.format(output))

if __name__== "__main__":
    app.run(debug=True)
    