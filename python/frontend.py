import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="💔",
    layout="centered"
)

# LOADING THE MODELS
with open('./MLYoung_pickle.obj', 'rb') as f:
    classifier_y = pickle.load(f)

with open('./MLOld_pickle.obj', 'rb') as f:
    classifier_o = pickle.load(f)

    
# HEADER SECTION
headerContainer = st.container()
headerText = """
    _A Demonstration of Machine Learning in Medicine_
"""
headerDescription = """
    We have created a model that accurately predicts the likelihood of a person having heart disease.
"""
with headerContainer:
    st.title("""
        💔 Heart Disease Predictor 💔
        ---
    """)
    st.subheader(headerText)
    st.markdown(headerDescription)

# DATASET SECTION
datasetContainer = st.container()

datasetTitle = "📊 Dataset 📊"

datasetLink = "https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download"
datasetText = """
    Our training [dataset]({})
    features 14 attributes
    (age, sex, blood pressure, cholesterol, etc).
""".format(datasetLink)

datasetExpander = st.expander("_Show Preview_")
datasetCaption = """
    _Here's a preview of the first five data points..._
"""
with datasetContainer:
    st.header(datasetTitle)
    st.markdown(datasetText)
    with datasetExpander:
        st.caption(datasetCaption)
        df = pd.read_csv('./heart.csv')
        st.table(df.head())

# MODEL SECTION
modelContainer = st.container()
with modelContainer:
    st.header("🩺 Model 🩺")
    st.markdown("Input your data as well and we'll be able to give you a prediction as to whether you're at risk of a heart attack or not!")
    age = st.number_input("How old are you?", min_value=0, max_value=200, step=1, format="%d")
    # age = st.slider("How old are you?", min_value=0, max_value=125, step=1, format="%d")

    # sex = st.number_input('Male (1) or Female (0) ?')
    sexRaw = st.radio("Gender", options=["Male", "Female"])
    sex = 1 if sexRaw == "Male" else 0
    
    cp = st.number_input('What type of chest pain do you have? (1) typical angina (2) atypical angina (3) non-anginal pain (4) asymptomatic')
    trt = st.number_input('What is resting blood pressure (in mm Hg)')
    chol = st.number_input('What is your cholesterol in mg/dl fetched via BMI sensor')
    fbs = st.number_input('Is your fasting blood sugar over 120mg/dl? ( 1 / 0 )')
    restecg = st.number_input('What is your resting electrocardiographic results? (0) Normal? (1)  having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) (2) showing probable or definite left ventricular hypertrophy by Estes criteria')
    thalachh = st.number_input('What is your maximum heart rate achieved')
    exng = st.number_input('When you exercise do you have induced angina? (1) Yes? (0) No?')
    oldpeak = st.number_input('What is your previous peak?')
    slp = st.number_input('What is the slope of that peak')
    caa = st.number_input('How many major vessels do you have?')
    thall = st.number_input('What were the results of your Thallium Stress Test Results? (0 - 3)')

    user_data = [[age, sex, cp, trt, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]]
    prediction = ''
    prediction_percentage = ''

    if st.button('Predict'):
        if age >= 50:
            # run predict 
            prediction = classifier_o.predict(user_data)
            print(prediction)
            prediction_percentage = classifier_o.predict_proba(user_data)
            print(prediction_percentage)
            # get the index with the highest percentage
            prediction_ind = np.argmax(prediction_percentage)
            # get the index of positive, so we can later display the percentage of positive.
            positive_ind = np.where(prediction[0] == 1)
            prediction = 'Positive' if prediction[0][prediction_ind]==1 else 'Negative'
            prediction_percentage = prediction_percentage[0][positive_ind[0]][0]
        else:
            # run predict 
            prediction = classifier_y.predict(user_data)
            print(prediction)
            prediction_percentage = classifier_o.predict_proba(user_data)
            print(prediction_percentage)
            # get the index with the highest percentage
            prediction_ind = np.argmax(prediction_percentage)
            # get the index of positive, so we can later display the percentage of positive.
            positive_ind = np.where(prediction[0] == 1)
            prediction = 'Positive' if prediction[0][prediction_ind]==1 else 'Negative'
            prediction_percentage = prediction_percentage[0][positive_ind[0]][0]

# RESULT SECTION
resultContainer = st.container()
with resultContainer:
    if prediction:
        st.title(f"Your prediction is: {prediction}")
        st.markdown(f"You have a **{round(prediction_percentage*100, 2)}%** chance of having a heart disease.")