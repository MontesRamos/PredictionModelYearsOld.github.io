import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return make_response(
        jsonify(
            username="ruben",
            surname="montes",
            day="now",
            idOperacion="123123",
            pais="Peru",
            parametro1="parametro 1",
            parametro2="parametro 2",
            prediction=output
        ),200
    )
    '''return {'prediction': output}'''
    '''return render_template('index.html', prediction_text='La predicción obtenida es... $ {}'.format(output))'''

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)