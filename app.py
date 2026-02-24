from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

with open('salary_model2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

job_titles = list(encoders['job_title'].classes_)
locations = list(encoders['location'].classes_)
experiences = list(encoders['experience'].classes_)
employments = list(encoders['employment'].classes_)
sizes = list(encoders['size'].classes_)

german_countries = ['Germany', 'France', 'Spain']

@app.route('/')
def home():
    return render_template('index.html',
                           job_titles=job_titles,
                           locations=locations,
                           experiences=experiences,
                           employments=employments,
                           sizes=sizes)

@app.route('/predict', methods=['POST'])
def predict():
    work_year = 2024
    job_title = request.form['job_title']
    location = request.form['location']
    experience = request.form['experience']
    employment = request.form['employment']
    size = request.form['size']

    job_encoded = encoders['job_title'].transform([job_title])[0]
    location_encoded = encoders['location'].transform([location])[0]
    experience_encoded = encoders['experience'].transform([experience])[0]
    employment_encoded = encoders['employment'].transform([employment])[0]
    size_encoded = encoders['size'].transform([size])[0]

    input_data = pd.DataFrame([[work_year, job_encoded, location_encoded,
                                experience_encoded, employment_encoded, size_encoded]],
                              columns=['work_year', 'job_encoded', 'location_encoded',
                                      'experience_encoded', 'employment_encoded', 'size_encoded'])

    prediction = model.predict(input_data)[0]

    if location in german_countries:
        currency = "€"
    elif location == 'United Kingdom':
        currency = "£"
    else:
        currency = "$"

    salary = f"{currency}{prediction:,.0f}"

    return render_template('index.html',
                           prediction=salary,
                           job_title=job_title,
                           location=location,
                           experience=experience,
                           employment=employment,
                           size=size,
                           job_titles=job_titles,
                           locations=locations,
                           experiences=experiences,
                           employments=employments,
                           sizes=sizes)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
