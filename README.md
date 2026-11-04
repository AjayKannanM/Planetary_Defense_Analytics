# 🌠 Planetary Defense Analytics: Asteroid Intelligence & Hazard Explorer

An end-to-end machine learning web application built to analyze near-Earth objects, classify potentially hazardous asteroids (PHAs), predict asteroid diameters, and cluster astronomical populations using advanced tree-based models and interactive visualizations.

## 🚀 Deployed Application
Access the live interactive Streamlit application here:
👉 [Planetary Defense Analytics App](https://planetarydefenseanalytics-4dqh5jnhhe5pbaeavwbkrz.streamlit.app/)

---

## 📋 Cases Completed
1. **PHA Classification (Classification Case):** Trained and optimized classification models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, CatBoost) handling target imbalance to identify potentially hazardous asteroids.
2. **Diameter Regression (Regression Case):** Built regression pipelines to estimate physical asteroid diameters accurately.
3. **Cluster Analysis (Clustering Case):** Unsupervised grouping of astronomical populations with interactive feature-space visualization.
4. **Explainable AI (XAI):** Integrated SHAP waterfall plots for real-time local model explainability on user inputs.

---

## 📂 Project Structure

text
Planetary_Defense_Analytics/
│
├── models/                  # Trained ML pipelines (tracked via Git LFS)
├── notebooks/               # Jupyter notebooks for EDA, cleaning, and model building
├── app.py                   # Streamlit web application script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

---

## 💻 How to Run Locally

1. **Clone the repository:**

bash
git clone https://github.com/AjayKannanM/Planetary_Defense_Analytics.git
cd Planetary_Defense_Analytics

2. **Install dependencies:**    

bash
pip install -r requirements.txt

3. **Run the Streamlit application:**

bash
streamlit run app.py