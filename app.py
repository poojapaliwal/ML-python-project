import pickle, scipy
from flask import Flask, jsonify, request
import numpy as np
import joblib

model = pickle.load(open('model1.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "hello world"


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age')
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalach = request.form.get('thalach')
    exang = request.form.get('exang')
    oldpeak = request.form.get('oldpeak')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')
    print(type(age))
    input_query = np.array([[int(age), int(sex), int(cp), int(trestbps), int(chol),
                             int(fbs), int(restecg), int(thalach)
                                , int(exang), int(oldpeak), int(slope), int(ca), int(thal)]])

    input_query = input_query.reshape(1, -1)
    result = model.predict(input_query)
    final_result = result[0]

    return jsonify({'Disease: ': str(final_result)})


if __name__ == '__main__':
    app.run(debug=True)
