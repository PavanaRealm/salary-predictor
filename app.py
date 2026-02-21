from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model
with open('salary_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = float(request.form['experience'])
    prediction = model.predict(pd.DataFrame({'years_experience': [experience]}))[0]
    salary = f"${prediction:,.0f}"
    return render_template('index.html', prediction=salary, experience=experience)

if __name__ == '__main__':
    app.run(debug=True)