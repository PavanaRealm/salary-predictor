# 💰 Salary Predictor - ML Web App

An end-to-end Machine Learning web app that predicts salary based on years of experience, job title, location and education level.

## 🚀 Live Demo
👉 https://salary-predictor-vkmt.onrender.com

## 🤖 Algorithm Used
Linear Regression — chosen for its interpretability and effectiveness at predicting continuous values like salary. The model learns the relationship between input features and salary from historical data.

## 📊 Model Performance
- R² Score: 0.9810 (98.10% accurate)
- Mean Absolute Error: ~$5,500

## ✅ Features
- Predicts salaries for US and German cities
- Smart currency detection — Euro for Germany, Dollar for US
- Cities: San Francisco, New York, Berlin, Munich, Hamburg, Frankfurt
- Deployed live on the internet

## 🛠️ Tech Stack
- Python — Core programming language
- Scikit-learn — Machine learning model
- Pandas and NumPy — Data manipulation
- Matplotlib and Seaborn — Data visualization
- Flask — Web framework
- HTML and CSS — Frontend

## ⚙️ How to Run Locally
1. Clone the repository: git clone https://github.com/PavanaRealm/salary-predictor.git
2. Install dependencies: pip install -r requirements.txt
3. Run the app: python3 app.py
4. Open browser and go to http://127.0.0.1:5000

## 👩‍💻 Author
Pavani Bhuvaneswari
Aspiring Data Scientist | ML Enthusiast


## Model Transparency
View the full model card with bias analysis and performance metrics:
- Live: https://salary-predictor-vkmt.onrender.com/model-card
- Includes: R² score, MAE, RMSE by country and experience level
- Known limitations and data bias documented
