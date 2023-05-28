import streamlit as st
import pickle
import pandas as pd
import operator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold   
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

header = st.container()
dataset = st.container()
model_training = st.container()
results = st.container()

with open('LogRegr_pickle.obj', 'rb') as f:
    classifier = pickle.load(f)

with header:
    st.title("Group 30: Heart-disease predictor")
    st.text("Our group's project is to be able to create a model")
    st.text("that will accurately determine whether a person is at risk")
    st.text("of suffering a heart-attack")
    
with dataset:
    st.header("Heart Disease Dataset")
    st.text("We found this dataset on Kaggle.com, and it features 14 attributes that range from")
    st.text("a person's age, sex, blood pressure, cholesterol, and more. Here is the dataset")

    df = pd.read_csv('./heart.csv')
    st.write(df.head())

with model_training:
    st.header("Heart Disease Model")
    st.text("Input your data as well and we'll be able to give you a prediction as to whether you're at risk of heart disease or not!")
    age = st.number_input("How old are you?")
    sex = st.number_input('Male (1) or Female (0) ?')
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

    if st.button('Predict'):
        prediction = classifier.predict(user_data)
        prediction = 'Positive' if prediction[0]==1 else 'Negative'

with results:
    if prediction:
        st.title(f"Your prediction is: {prediction}")