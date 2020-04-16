import flask
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import run_model
from IPython.core.debugger import set_trace
from model import dummy_model,run_model,smart_model

# App definition
app = Flask(__name__,template_folder='templates')


@app.route('/')
def home():
    return render_template('../index.html')

@app.route('/trainmodel',methods=['GET'])
def trainmodel():
    try:
        run_model()
        resp = jsonify(success=True)
        resp.status_code = 200
        return resp
    except:
        resp = jsonify(success=False)
        resp.status_code = 201
        return jsonify(resp)
    
@app.route('/predictGUI',methods=['POST'])
def predictGUI():
    file_name = request.form.get('filename')
    prediction = dummy_model(file_name)

    return render_template('index.html', emotion_pred='Emotion predicted {}'.format(prediction))

@app.route('/predict',methods=['POST'])
def predict():
    try:
        file_name = request.args.get('filename')
        prediction = dummy_model(file_name)
        return jsonify(prediction)
    except:
        #TODO: handle better the exceptions s.t. a clean messages displays in PostMan
        resp = jsonify(success=False)
        resp.status_code = 201
        return jsonify(resp)
    
@app.route('/predictJSON',methods=['POST'])
def predictJSON():
    try:
        data = request.get_json(force=True)
        prediction = {val[0]:dummy_model(val[1]) for val in data.items()}

        return jsonify(prediction)
    except:
        resp = jsonify(success=False)
        resp.status_code = 201
        return jsonify(resp)

if __name__ == "__main__":
    app.run(debug=True,host= '0.0.0.0',port=5000)