from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

with open('salary_model2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

job_titles = list(encoders['job_title'].classes_)
locations = list(encoders['location'].classes_)
educations = list(encoders['education'].classes_)

german_cities = ['Berlin', 'Munich', 'Hamburg', 'Frankfurt']

@app.route('/')
def home():
    return render_template('index.html',
                           job_titles=job_titles,
                           locations=locations,
                           educations=educations)

@app.route('/predict', methods=['POST'])
def predict():
    experience = float(request.form['experience'])
    job_title = request.form['job_title']
    location = request.form['location']
    education = request.form['education']

    job_encoded = encoders['job_title'].transform([job_title])[0]
    location_encoded = encoders['location'].transform([location])[0]
    education_encoded = encoders['education'].transform([education])[0]

    input_data = pd.DataFrame([[experience, job_encoded, location_encoded, education_encoded]],
                              columns=['years_experience', 'job_title_encoded', 'location_encoded', 'education_encoded'])

    prediction = model.predict(input_data)[0]
    currency = "€" if location in german_cities else "$"
    salary = f"{currency}{prediction:,.0f}"

    return render_template('index.html',
                           prediction=salary,
                           experience=experience,
                           job_title=job_title,
                           location=location,
                           education=education,
                           job_titles=job_titles,
                           locations=locations,
                           educations=educations)

if __name__ == '__main__':
    app.run(debug=True)
