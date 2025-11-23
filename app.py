import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI Health Assistant",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
            
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 20px;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 2px solid #28a745;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading models and scalers
diabetes_model = pickle.load(open(f'{working_dir}/diabetes/diabetes_model.sav', 'rb'))
diabetes_scaler = pickle.load(open(f'{working_dir}/diabetes/diabetes_scaler.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/heart/heart_model.sav', 'rb'))
heart_scaler = pickle.load(open(f'{working_dir}/heart/heart_scaler.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons/parkinsons_model.sav', 'rb'))
parkinsons_scaler = pickle.load(open(f'{working_dir}/parkinsons/parkinsons_scaler.sav', 'rb'))

# Sidebar navigation with enhanced styling
with st.sidebar:
    st.markdown("### 🏥 AI Health Prediction System")
    st.markdown("---")
    
    selected = option_menu(
        'Choose Disease Prediction',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['droplet-fill', 'heart-pulse-fill', 'activity'],
        menu_icon='hospital-fill',
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#262730"},
            "icon": {"color": "white", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )
    
    st.markdown("---")
    st.markdown("💡 **Enter patient details** and click the prediction button to get results.")
    st.markdown("✅ All predictions are based on Machine Learning models.")

# DIABETES PREDICTION 
if selected == 'Diabetes Prediction':
    
    st.markdown("<h1>🩸 Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("ℹ️ About Diabetes Prediction", expanded=False):
        st.write("""
        This tool predicts the likelihood of diabetes based on various health parameters.
        Enter the patient's medical data below to get an instant prediction.
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('👶 Number of Pregnancies', min_value=0, max_value=20, value=0)
        SkinThickness = st.number_input('📏 Skin Thickness (mm)', min_value=0, max_value=100, value=20)
        DiabetesPedigreeFunction = st.number_input('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    
    with col2:
        Glucose = st.number_input('🍬 Glucose Level (mg/dL)', min_value=0, max_value=300, value=120)
        Insulin = st.number_input('💉 Insulin Level (μU/mL)', min_value=0, max_value=900, value=100)
        Age = st.number_input('🎂 Age (years)', min_value=1, max_value=120, value=30)
    
    with col3:
        BloodPressure = st.number_input('❤️ Blood Pressure (mmHg)', min_value=0, max_value=200, value=80)
        BMI = st.number_input('⚖️ BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    
    if st.button('🔍 Predict Diabetes Risk'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            user_input = np.array(user_input).reshape(1, -1)
            user_input_scaled = diabetes_scaler.transform(user_input)
            diab_prediction = diabetes_model.predict(user_input_scaled)
            
            if diab_prediction[0] == 1:
                st.markdown("<div class='warning-box'>⚠️ HIGH RISK: The person is likely to have diabetes<br>🩺 Consult a healthcare professional immediately</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>✅ LOW RISK: The person is unlikely to have diabetes<br>💚 Maintain a healthy lifestyle!</div>", unsafe_allow_html=True)
        except:
            st.error('❌ Please enter valid numbers for all fields')

# HEART DISEASE PREDICTION 
elif selected == 'Heart Disease Prediction':
    
    st.markdown("<h1>❤️ Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("ℹ️ About Heart Disease Prediction", expanded=False):
        st.write("""
        This tool predicts the risk of heart disease based on cardiac health parameters.
        Provide accurate medical information for reliable predictions.
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('🎂 Age', min_value=1, max_value=120, value=50)
        cp = st.selectbox('💔 Chest Pain Type', [0, 1, 2, 3], format_func=lambda x: ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][x])
        fbs = st.selectbox('🍬 Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        exang = st.selectbox('🏃 Exercise Induced Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
        ca = st.selectbox('🔬 Major Vessels (0-3)', [0, 1, 2, 3])
    
    with col2:
        sex = st.selectbox('👤 Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
        trestbps = st.number_input('❤️ Resting Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)
        restecg = st.selectbox('📊 Resting ECG', [0, 1, 2], format_func=lambda x: ['Normal', 'ST-T Abnormality', 'LV Hypertrophy'][x])
        oldpeak = st.number_input('📉 ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        thal = st.selectbox('🫀 Thalassemia', [0, 1, 2], format_func=lambda x: ['Normal', 'Fixed Defect', 'Reversible Defect'][x])
    
    with col3:
        chol = st.number_input('🧪 Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
        thalach = st.number_input('💓 Max Heart Rate', min_value=60, max_value=220, value=150)
        slope = st.selectbox('📈 Slope of Peak Exercise ST', [0, 1, 2], format_func=lambda x: ['Upsloping', 'Flat', 'Downsloping'][x])
    
    if st.button('🔍 Predict Heart Disease Risk'):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = np.array(user_input).reshape(1, -1)
            user_input_scaled = heart_scaler.transform(user_input)
            heart_prediction = heart_disease_model.predict(user_input_scaled)
            
            if heart_prediction[0] == 1:
                st.markdown("<div class='warning-box'>⚠️ HIGH RISK: Heart disease detected<br>🩺 Immediate medical consultation recommended</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>✅ LOW RISK: No heart disease detected<br>💚 Keep up the healthy heart habits!</div>", unsafe_allow_html=True)
        except:
            st.error('❌ Please enter valid data for all fields')

# PARKINSON'S PREDICTION 
elif selected == "Parkinsons Prediction":
    
    st.markdown("<h1>🧠 Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.expander("ℹ️ About Parkinson's Prediction", expanded=False):
        st.write("""
        This tool analyzes voice measurements to predict Parkinson's disease.
        Voice features are used as biomarkers for early detection.
        """)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', value=150.0)
        RAP = st.number_input('MDVP:RAP', value=0.002)
        APQ3 = st.number_input('Shimmer:APQ3', value=0.01)
        NHR = st.number_input('NHR', value=0.02)
        spread1 = st.number_input('spread1', value=-5.0)
    
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', value=200.0)
        PPQ = st.number_input('MDVP:PPQ', value=0.002)
        APQ5 = st.number_input('Shimmer:APQ5', value=0.01)
        HNR = st.number_input('HNR', value=20.0)
        spread2 = st.number_input('spread2', value=0.2)
    
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', value=100.0)
        DDP = st.number_input('Jitter:DDP', value=0.006)
        APQ = st.number_input('MDVP:APQ', value=0.02)
        RPDE = st.number_input('RPDE', value=0.5)
        D2 = st.number_input('D2', value=2.0)
    
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.005)
        Shimmer = st.number_input('MDVP:Shimmer', value=0.03)
        DDA = st.number_input('Shimmer:DDA', value=0.03)
        DFA = st.number_input('DFA', value=0.7)
        PPE = st.number_input('PPE', value=0.2)
    
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', value=0.00003)
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', value=0.3)
    
    if st.button("🔍 Predict Parkinson's Disease"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, 
                         APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            user_input = np.array(user_input).reshape(1, -1)
            user_input_scaled = parkinsons_scaler.transform(user_input)
            parkinsons_prediction = parkinsons_model.predict(user_input_scaled)
            
            if parkinsons_prediction[0] == 1:
                st.markdown("<div class='warning-box'>⚠️ POSITIVE: Parkinson's disease detected<br>🩺 Consult a neurologist for confirmation</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>✅ NEGATIVE: No Parkinson's disease detected<br>💚 Voice parameters are within normal range</div>", unsafe_allow_html=True)
        except:
            st.error('❌ Please enter valid numbers for all voice parameters')

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>🏥 <b>AI-Powered Multiple Disease Prediction System</b></p>
        <p>⚠️ Disclaimer: This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
    </div>
""", unsafe_allow_html=True)
