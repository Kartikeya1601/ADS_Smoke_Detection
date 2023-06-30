from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final = np.array(float_features, ndmin=2)
    prediction = model.predict(final)
    res = str(prediction[0])
    if res == '0':
        ans = "not detected"
    else:
        ans = "detected"

    return render_template("test.html", y="Smoke is {}.".format(ans))


if __name__ == '__main__':
    app.run(debug=True, port=5100)
