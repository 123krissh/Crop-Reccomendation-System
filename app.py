from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained models
model1 = pickle.load(open("models/model1.pkl", "rb"))
model2 = pickle.load(open("models/model2.pkl", "rb"))
model3 = pickle.load(open("models/model3.pkl", "rb"))
models = [model1, model2, model3]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            print("Form data received:", request.form)
            # Get form input
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            input_data = [[N, P, K, temp, humidity, ph, rainfall]]

            # Predict probabilities from all models
            probas = [model.predict_proba(input_data) for model in models]
            avg_proba = np.mean(probas, axis=0)

            crop_classes = models[0].classes_
            top_index = np.argmax(avg_proba[0])  # Only top 1
            top_crop = (crop_classes[top_index], round(avg_proba[0][top_index]*100, 2))

            return render_template('index.html', top_crop=top_crop)

        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Hello, Flask!"

# if __name__ == '__main__':
#     app.run(debug=True)

