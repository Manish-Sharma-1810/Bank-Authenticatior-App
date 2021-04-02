from flask import Flask,request, url_for, redirect, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# load the pickle file and open it in readbyte mode

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict_note_authentication():
    '''
    For rendering results on HTML GUI

    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = classifier.predict(final_features)

    return render_template('index.html', predicted_value=' Predicted value is : {}'.format(prediction[0]))



if __name__== '__main__':
    app.run(debug=True)
