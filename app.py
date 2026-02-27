from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os
from groq import Groq
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

app = Flask(__name__)

with open('salary_model3.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)

df_raw = pd.read_csv('jobs_in_data.csv')
job_mapping = feature_info['job_mapping']
df_raw['job_title_normalized'] = df_raw['job_title'].map(job_mapping)
df_real = df_raw.copy()

feature_columns = feature_info['columns']
size_map = feature_info['size_map']
exp_map = feature_info['exp_map']
top_locations = feature_info['top_locations']
top_jobs = sorted(feature_info['top_jobs'])

job_titles = top_jobs
locations = top_locations
experiences = list(exp_map.keys())
employments = ['Full-time', 'Part-time', 'Contract', 'Freelance']
sizes = ['S', 'M', 'L']

# Currency mapping - all eurozone countries get euro
euro_countries = ['Germany', 'France', 'Spain', 'Portugal', 'Netherlands', 'Greece', 'Italy']
pound_countries = ['United Kingdom']
dollar_countries = ['United States', 'Canada', 'Australia']

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

def get_currency(location):
    if location in euro_countries:
        return 'euro'
    elif location in pound_countries:
        return 'pound'
    return 'dollar'

def format_salary(value, currency):
    symbols = {'euro': '€', 'pound': '£', 'dollar': '$'}
    return symbols[currency] + '{:,.0f}'.format(int(value))

def build_input(job_title, location, experience, employment, size):
    row = {col: 0 for col in feature_columns}
    row['work_year'] = 2024
    row['size_encoded'] = size_map.get(size, 1)
    row['experience_encoded'] = exp_map.get(experience, 1)
    job_col = 'job_title_normalized_' + job_title
    loc_col = 'company_location_' + location
    emp_col = 'employment_type_' + employment
    if job_col in row: row[job_col] = 1
    if loc_col in row: row[loc_col] = 1
    if emp_col in row: row[emp_col] = 1
    return pd.DataFrame([row])

def predict_salary(job_title, location, experience, employment, size):
    try:
        input_data = build_input(job_title, location, experience, employment, size)
        prediction = model.predict(input_data)[0]
        currency = get_currency(location)
        filtered = df_real[
            (df_real['job_title_normalized'] == job_title) &
            (df_real['company_location'] == location) &
            (df_real['experience_level'] == experience) &
            (df_real['salary_in_usd'] > 20000) &
            (df_real['salary_in_usd'] < 400000)
        ]['salary_in_usd']
        if len(filtered) > 5:
            min_val = float(filtered.quantile(0.1))
            max_val = float(filtered.quantile(0.9))
        else:
            min_val = prediction * 0.8
            max_val = prediction * 1.2
        return {
            'prediction': format_salary(prediction, currency),
            'min_salary': format_salary(min_val, currency),
            'max_salary': format_salary(max_val, currency),
            'records_used': len(filtered),
            'low_data': len(filtered) < 10
        }
    except Exception as e:
        print('Prediction error:', e)
        return None

def get_chart_data(job_title=None, experience=None):
    df_f = df_real[df_real['company_location'].isin(top_locations) & df_real['job_title_normalized'].notna()]
    df_f = df_f[(df_f['salary_in_usd'] > 20000) & (df_f['salary_in_usd'] < 400000)]
    df_loc = df_f[df_f['job_title_normalized'] == job_title] if job_title else df_f
    loc_label = 'Avg salary for ' + str(job_title) + ' by country' if job_title else 'Avg salary by country (all job titles)'
    df_job = df_f[df_f['experience_level'] == experience] if experience else df_f
    job_label = 'Avg salary by job title (experience: ' + str(experience) + ')' if experience else 'Avg salary by job title (all levels)'
    loc_avg = df_loc.groupby('company_location')['salary_in_usd'].mean().reindex(top_locations).dropna()
    job_avg = df_job.groupby('job_title_normalized')['salary_in_usd'].mean().reindex(top_jobs).dropna().sort_values(ascending=False)
    return {
        'location': {'labels': list(loc_avg.index), 'values': [int(v) for v in loc_avg.values], 'label': loc_label},
        'jobs': {'labels': list(job_avg.index), 'values': [int(v) for v in job_avg.values], 'label': job_label}
    }

@app.route('/')
def home():
    chart_data = get_chart_data()
    return render_template('index.html', job_titles=job_titles, locations=locations,
                           experiences=experiences, employments=employments, sizes=sizes,
                           chart_data=chart_data)

@app.route('/predict', methods=['POST'])
def predict():
    job_title = request.form['job_title']
    location = request.form['location']
    experience = request.form['experience']
    employment = request.form['employment']
    size = request.form['size']
    result = predict_salary(job_title, location, experience, employment, size)
    chart_data = get_chart_data(job_title=job_title, experience=experience)
    if result:
        return render_template('index.html',
                               prediction=result['prediction'],
                               min_salary=result['min_salary'],
                               max_salary=result['max_salary'],
                               records_used=result['records_used'],
                               low_data=result['low_data'],
                               job_title=job_title, location=location,
                               experience=experience, employment=employment, size=size,
                               job_titles=job_titles, locations=locations,
                               experiences=experiences, employments=employments, sizes=sizes,
                               chart_data=chart_data)
    return render_template('index.html', job_titles=job_titles, locations=locations,
                           experiences=experiences, employments=employments, sizes=sizes,
                           chart_data=chart_data)

@app.route('/model-card')
def model_card():
    df_m = df_real[df_real['company_location'].isin(top_locations) & df_real['job_title_normalized'].notna()].copy()
    df_m = df_m[(df_m['salary_in_usd'] > 20000) & (df_m['salary_in_usd'] < 400000)]
    df_m['size_encoded'] = df_m['company_size'].map(size_map)
    df_m['experience_encoded'] = df_m['experience_level'].map(exp_map)
    df_encoded = pd.get_dummies(df_m[['job_title_normalized', 'company_location', 'employment_type']], drop_first=False)
    X = pd.concat([df_m[['work_year', 'size_encoded', 'experience_encoded']].reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)
    for col in feature_columns:
        if col not in X.columns: X[col] = 0
    X = X[feature_columns]
    y = df_m['salary_in_usd'].reset_index(drop=True)
    preds = model.predict(X)
    df_m = df_m.reset_index(drop=True)
    df_m['predicted'] = preds
    country_metrics = []
    for loc in top_locations:
        subset = df_m[df_m['company_location'] == loc]
        if len(subset) > 5:
            mae = mean_absolute_error(subset['salary_in_usd'], subset['predicted'])
            r2 = r2_score(subset['salary_in_usd'], subset['predicted'])
            country_metrics.append({'country': loc, 'records': len(subset), 'mae': int(mae), 'r2': round(r2, 3)})
    exp_metrics = []
    for exp in ['Entry-level', 'Mid-level', 'Senior', 'Executive']:
        subset = df_m[df_m['experience_level'] == exp]
        if len(subset) > 5:
            mae = mean_absolute_error(subset['salary_in_usd'], subset['predicted'])
            r2 = r2_score(subset['salary_in_usd'], subset['predicted'])
            exp_metrics.append({'experience': exp, 'records': len(subset), 'mae': int(mae), 'r2': round(r2, 3)})
    metrics = {
        'overall': {'r2': round(r2_score(y, preds), 4), 'mae': int(mean_absolute_error(y, preds)), 'rmse': int(np.sqrt(mean_squared_error(y, preds)))},
        'by_country': country_metrics,
        'by_experience': exp_metrics,
        'data_split': {'total': len(df_m)},
        'data_distribution': {'United States': int(df_real[df_real['company_location'] == 'United States'].shape[0])}
    }
    return render_template('model_card.html', metrics=metrics)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    system_prompt = 'You are a salary prediction assistant. Extract job details from the user message. Available: Job Titles: ' + str(job_titles) + ', Locations: ' + str(locations) + ', Experience: ' + str(list(exp_map.keys())) + ', Employment: ' + str(employments) + ', Size: S=Small M=Medium L=Large. Return ONLY JSON: {"job_title": "Data Scientist", "location": "Germany", "experience": "Senior", "employment": "Full-time", "size": "M", "ready": true} or {"ready": false, "message": "Please provide all details."}'
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_message}]
    )
    result = json.loads(response.choices[0].message.content)
    if result.get('ready'):
        salary_result = predict_salary(result['job_title'], result['location'], result['experience'], result['employment'], result['size'])
        if salary_result:
            jt = result['job_title']
            loc = result['location']
            exp = result['experience']
            pred = salary_result['prediction']
            mn = salary_result['min_salary']
            mx = salary_result['max_salary']
            warning = ' Warning: Limited data for this combination.' if salary_result['low_data'] else ''
            msg = 'Based on your profile as a ' + jt + ' in ' + loc + ' with ' + exp + ' experience, your predicted salary is ' + pred + ' (Range: ' + mn + ' - ' + mx + ')' + warning
            return jsonify({'response': msg})
        else:
            return jsonify({'response': 'Sorry I could not predict the salary. Please check your inputs.'})
    else:
        return jsonify({'response': result.get('message')})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
