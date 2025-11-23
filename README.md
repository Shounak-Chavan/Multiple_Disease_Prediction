<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" />
  <img src="https://img.shields.io/github/repo-size/Shounak-Chavan/Multiple_Disease_Prediction" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>


# 🏥 AI-Powered Multiple Disease Prediction System
### *(Diabetes • Heart Disease • Parkinson’s)*

This project is a **Machine Learning–based health prediction system** that predicts the risk of three major diseases — **Diabetes**, **Heart Disease**, and **Parkinson’s Disease** — using trained classification models and an interactive **Streamlit** web app.

The system takes medical input parameters from users, processes them using **scalers**, and produces predictions instantly.

---

## 🚀 Features

- 🔍 **Diabetes Risk Prediction** using 8 clinical parameters  
- ❤️ **Heart Disease Prediction** using 13 medical features  
- 🧠 **Parkinson’s Disease Prediction** using voice-related biomarkers  
- 🎨 Beautiful **Streamlit UI** with sidebar navigation  
- ⚙️ Each disease has **its own model, scaler, and dataset**  
- 📂 Fully modular & clean folder structure  
- 🛡 Professional `.gitignore` included  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- NumPy  
- Pandas  
- Scikit-Learn  
- Pickle  

---

## 📁 Project Structure

```
Multiple_Disease_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── Diabetes/
│   ├── diabetes_model.sav
│   ├── diabetes_scaler.sav
│   ├── diabetes.csv
│   └── DiabetesPred.ipynb
│
├── Heart/
│   ├── heart_model.sav
│   ├── heart_scaler.sav
│   ├── heart.csv
│   └── HeartDiseasesPred.ipynb
│
└── Parkinsons/
    ├── parkinsons_model.sav
    ├── parkinsons_scaler.sav
    ├── parkinsons.data
    └── ParkinsonsDetect.ipynb
```

---

## ▶️ How to Run Locally

### **1️⃣ Clone the Repository**
```
git clone https://github.com/Shounak-Chavan/Multiple_Disease_Prediction.git
cd Multiple_Disease_Prediction
```

### **2️⃣ (Optional) Create Virtual Environment**
```
python -m venv venv
```

Activate the venv:

**Windows:**
```
venv\Scripts\activate
```

**Mac/Linux:**
```
source venv/bin/activate
```

### **3️⃣ Install Libraries**
```
pip install -r requirements.txt
```

### **4️⃣ Run Streamlit App**
```
streamlit run app.py
```

---

## 🧠 Models Used

### Diabetes:
- Logistic Regression / SVM / XGBoost / CatBoost / LightGBM

### Heart Disease:
- Logistic Regression / SVM / XGBoost / CatBoost / LightGBM

### Parkinson’s:
- Logistic Regression / SVM / XGBoost / CatBoost / LightGBM

---

## 📌 Future Improvements

- Add more diseases (Kidney, Liver, Breast Cancer etc.)  
- Deploy on Streamlit Cloud / Render / HuggingFace  
- Add login system  
- Add charts and visualization  
- Make UI more interactive  

---

## ⚠️ Disclaimer
This project is **for educational purposes only** and should not be used for real medical diagnosis.

---

## ⭐ If you found this project useful, please ⭐ the repository!
