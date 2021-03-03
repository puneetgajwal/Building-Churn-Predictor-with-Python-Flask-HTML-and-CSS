from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__, template_folder='template')
adaboost = pickle.load(open('adaboost.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("homepage.html")

def get_data():
    tenure = request.form.get('tenure')
    MonthlyCharges = request.form.get('MonthlyCharges')
    TotalCharges = request.form.get('TotalCharges')
    gender = request.form.get('gender')
    SeniorCitizen = request.form.get('SeniorCitizen')
    Partner = request.form.get('Partner')
    Dependents = request.form.get('Dependents')
    PhoneService = request.form.get('PhoneService')
    MultipleLines = request.form.get('MultipleLines')
    InternetService = request.form.get('InternetService')
    OnlineSecurity = request.form.get('OnlineSecurity')
    OnlineBackup = request.form.get('OnlineBackup')
    DeviceProtection = request.form.get('DeviceProtection')
    TechSupport = request.form.get('TechSupport')
    StreamingTV = request.form.get('StreamingTV')
    StreamingMovies = request.form.get('StreamingMovies')
    Contract = request.form.get('Contract')
    PaperlessBilling = request.form.get('PaperlessBilling')
    PaymentMethod = request.form.get('PaymentMethod')

    d_dict = {'tenure': [tenure], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges], 'gender_Female': [0],
              'gender_Male': [0], 'SeniorCitizen_0': [0], 'SeniorCitizen_1': [0], 'Partner_No': [0],
              'Partner_Yes': [0], 'Dependents_No': [0], 'Dependents_Yes': [0], 'PhoneService_No': [0],
              'PhoneService_Yes': [0], 'MultipleLines_No': [0], 'MultipleLines_No phone service': [0],
              'MultipleLines_Yes': [0], 'InternetService_DSL': [0], 'InternetService_Fiber optic': [0],
              'InternetService_No': [0], 'OnlineSecurity_No': [0], 'OnlineSecurity_No internet service': [0],
              'OnlineSecurity_Yes': [0], 'OnlineBackup_No': [0], 'OnlineBackup_No internet service': [0],
              'OnlineBackup_Yes': [0], 'DeviceProtection_No': [0], 'DeviceProtection_No internet service': [0],
              'DeviceProtection_Yes': [0], 'TechSupport_No': [0], 'TechSupport_No internet service': [0],
              'TechSupport_Yes': [0], 'StreamingTV_No': [0], 'StreamingTV_No internet service': [0],
              'StreamingTV_Yes': [0], 'StreamingMovies_No': [0], 'StreamingMovies_No internet service': [0],
              'StreamingMovies_Yes': [0], 'Contract_Month-to-month': [0], 'Contract_One year': [0],
              'Contract_Two year': [0], 'PaperlessBilling_No': [0], 'PaperlessBilling_Yes':[0],
              'PaymentMethod_Bank transfer (automatic)': [0], 'PaymentMethod_Credit card (automatic)': [0],
              'PaymentMethod_Electronic check': [0], 'PaymentMethod_Mailed check': [0]}

    replace_list = [gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                    TechSupport, StreamingTV, StreamingMovies, Contract,
                    PaperlessBilling, PaymentMethod]

    for key, value in d_dict.items():
        if key in replace_list:
            d_dict[key] = 1


    return pd.DataFrame.from_dict(d_dict, orient='columns')

def feature_imp(model, data):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_30 = indices[:30]
    data = data.iloc[:, top_30]
    return data

def min_max_scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler.fit(data)
    data_scaled = scaler.fit_transform(data.values.reshape(30, -1))
    data = data_scaled.reshape(-1, 30)
    return pd.DataFrame(data)

@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    featured_data = feature_imp(adaboost, df)
    scaled_data = min_max_scale(featured_data)
    prediction = adaboost.predict(scaled_data)
    outcome = 'Churner'
    if prediction == 0:
        outcome = 'Non-Churner'

    return render_template('results.html', tables = [df.to_html(classes='data', header=True)],
                           result = outcome)



if __name__=="__main__":
    app.run(debug=True)