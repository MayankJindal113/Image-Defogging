from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('LogiModel.pkl','rb'))
# Loaded the ML Model in the app.py file    


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/upload',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('index.html',prediction='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),message="Call Fire brigade")
    else:
        return render_template('index.html',prediction='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),message="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run(debug=True)
