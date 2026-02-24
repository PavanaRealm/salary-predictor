from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os
from groq import Groq
import json

app = Flask(__name__)

with open("salary_model2.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

job_titles = list(encoders["job_title"].classes_)
locations = list(encoders["location"].classes_)
experiences = list(encoders["experience"].classes_)
employments = list(encoders["employment"].classes_)
sizes = list(encoders["size"].classes_)

german_countries = ["Germany", "France", "Spain"]
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def predict_salary(job_title, location, experience, employment, size):
    try:
        job_encoded = encoders["job_title"].transform([job_title])[0]
        location_encoded = encoders["location"].transform([location])[0]
        experience_encoded = encoders["experience"].transform([experience])[0]
        employment_encoded = encoders["employment"].transform([employment])[0]
        size_encoded = encoders["size"].transform([size])[0]
        input_data = pd.DataFrame([[2024, job_encoded, location_encoded, experience_encoded, employment_encoded, size_encoded]],
                                  columns=["work_year", "job_encoded", "location_encoded", "experience_encoded", "employment_encoded", "size_encoded"])
        prediction = model.predict(input_data)[0]
        if location in german_countries:
            currency = "euro"
        elif location == "United Kingdom":
            currency = "pound"
        else:
            currency = "dollar"
        symbols = {"euro": "€", "pound": "£", "dollar": "$"}
        return f"{symbols[currency]}{prediction:,.0f}"
    except Exception as e:
        return None

@app.route("/")
def home():
    return render_template("index.html", job_titles=job_titles, locations=locations,
                           experiences=experiences, employments=employments, sizes=sizes)

@app.route("/predict", methods=["POST"])
def predict():
    job_title = request.form["job_title"]
    location = request.form["location"]
    experience = request.form["experience"]
    employment = request.form["employment"]
    size = request.form["size"]
    salary = predict_salary(job_title, location, experience, employment, size)
    return render_template("index.html", prediction=salary, job_title=job_title,
                           location=location, experience=experience, employment=employment,
                           size=size, job_titles=job_titles, locations=locations,
                           experiences=experiences, employments=employments, sizes=sizes)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    system_prompt = f"""You are a salary prediction assistant. Extract job details from the user message.
Available options:
- Job Titles: {job_titles}
- Locations: {locations}
- Experience Levels: {experiences}
- Employment Types: {employments}
- Company Sizes: {sizes}
Return ONLY a JSON object like this:
{{"job_title": "Data Scientist", "location": "Germany", "experience": "Senior", "employment": "Full-time", "size": "M", "ready": true}}
If you cannot extract all details return:
{{"ready": false, "message": "Please tell me your job title, location, experience level, employment type and company size."}}"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    )
    result = json.loads(response.choices[0].message.content)
    if result.get("ready"):
        salary = predict_salary(result["job_title"], result["location"], result["experience"], result["employment"], result["size"])
        if salary:
            return jsonify({"response": f"Based on your profile as a {result['job_title']} in {result['location']} with {result['experience']} experience, your predicted salary is {salary}"})
        else:
            return jsonify({"response": "Sorry I could not predict the salary. Please check your inputs."})
    else:
        return jsonify({"response": result.get("message")})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
