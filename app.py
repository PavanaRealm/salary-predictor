from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os
from groq import Groq
import json
import numpy as np

app = Flask(__name__)

with open("salary_model2.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

df_real = pd.read_csv("jobs_in_data.csv")

job_titles = list(encoders["job_title"].classes_)
locations = list(encoders["location"].classes_)
experiences = list(encoders["experience"].classes_)
employments = list(encoders["employment"].classes_)
sizes = list(encoders["size"].classes_)

german_countries = ["Germany", "France", "Spain"]
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_currency(location):
    if location in german_countries:
        return "euro"
    elif location == "United Kingdom":
        return "pound"
    return "dollar"

def format_salary(value, currency):
    symbols = {"euro": "€", "pound": "£", "dollar": "$"}
    return f"{symbols[currency]}{value:,.0f}"

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
        currency = get_currency(location)

        top_locations = ["United States", "United Kingdom", "Canada", "Germany", "France", "Spain", "Australia"]
        top_jobs = ["Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer",
                    "Applied Scientist", "Research Scientist", "Analytics Engineer", "Data Architect"]
        filtered = df_real[
            (df_real["job_title"] == job_title) &
            (df_real["company_location"] == location) &
            (df_real["experience_level"] == experience) &
            (df_real["salary_in_usd"] > 20000) &
            (df_real["salary_in_usd"] < 400000)
        ]["salary_in_usd"]

        if len(filtered) > 5:
            min_val = filtered.quantile(0.1)
            max_val = filtered.quantile(0.9)
        else:
            min_val = prediction * 0.8
            max_val = prediction * 1.2

        return {
            "prediction": format_salary(prediction, currency),
            "min_salary": format_salary(min_val, currency),
            "max_salary": format_salary(max_val, currency),
        }
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
    result = predict_salary(job_title, location, experience, employment, size)
    if result:
        return render_template("index.html",
                               prediction=result["prediction"],
                               min_salary=result["min_salary"],
                               max_salary=result["max_salary"],
                               job_title=job_title, location=location,
                               experience=experience, employment=employment, size=size,
                               job_titles=job_titles, locations=locations,
                               experiences=experiences, employments=employments, sizes=sizes)
    return render_template("index.html", job_titles=job_titles, locations=locations,
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
        salary_result = predict_salary(result["job_title"], result["location"], result["experience"], result["employment"], result["size"])
        if salary_result:
            jt = result["job_title"]; loc = result["location"]; exp = result["experience"]
            pred = salary_result["prediction"]; mn = salary_result["min_salary"]; mx = salary_result["max_salary"]
            return jsonify({"response": f"Based on your profile as a {jt} in {loc} with {exp} experience, your predicted salary is {pred} (Range: {mn} - {mx})"})
        else:
            return jsonify({"response": "Sorry I could not predict the salary. Please check your inputs."})
    else:
        return jsonify({"response": result.get("message")})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
