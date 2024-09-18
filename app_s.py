import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd  
from sklearn.preprocessing import OneHotEncoder


Model = joblib.load("loan_approval_model.pkl")
Scaler = joblib.load("rb_scaler.pkl")


original_numerical_cols = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 
                           'residential_assets_value', 'commercial_assets_value', 
                           'luxury_assets_value', 'bank_asset_value']


numerical_cols = ['loan_amount', 'loan_term', 'cibil_score', 
                  'residential_assets_value', 'commercial_assets_value', 
                  'luxury_assets_value', 'bank_asset_value', 'income_to_loan_ratio', 
                  'loan_term_to_income_ratio']


cities = ['Ankara', 'Bursa', 'Erzurum', 'Istanbul', 'Izmir']


def feature_engineering(df):
    df['income_to_loan_ratio'] = df['loan_amount'] / df['income_annum']
    df['loan_term_to_income_ratio'] = df['loan_term'] / df['income_annum']
    df['loan_term'] = df['loan_term'] * 36
    df.drop(columns=['income_annum'], inplace=True)
    return df


def encode_city(df, city):
    # One-hot encoding for City
    city_encoder = OneHotEncoder(sparse_output=False, categories=[cities])
    city_encoded = city_encoder.fit_transform(df[['City']])
    city_encoded_df = pd.DataFrame(city_encoded, columns=[f"City_{c}" for c in cities])
    df = pd.concat([df, city_encoded_df], axis=1)
    df.drop(columns=['City'], inplace=True) 
    return df

def predict(no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score,
            residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, city):
    
    # Create DataFrame for input
    test_df = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [float(income_annum)],  # Will be dropped after feature engineering
        'loan_amount': [float(loan_amount)],
        'loan_term': [float(loan_term) * 365],
        'cibil_score': [float(cibil_score)],
        'residential_assets_value': [float(residential_assets_value)],
        'commercial_assets_value': [float(commercial_assets_value)],
        'luxury_assets_value': [float(luxury_assets_value)],
        'bank_asset_value': [float(bank_asset_value)],
        'City': [city]  # Add city for encoding
    })

    # Apply feature engineering
    test_df = feature_engineering(test_df)

    # Apply OneHotEncoding for City
    test_df = encode_city(test_df, city)

    # Scale only the numerical features that were scaled during training
    scaled_numerical_cols = ['loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value', 
                             'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    temp_df = pd.DataFrame(Scaler.transform(test_df[scaled_numerical_cols]), columns=scaled_numerical_cols)

    # Remove the original numerical columns and replace them with scaled ones
    test_df.drop(scaled_numerical_cols, axis=1, inplace=True)
    test_df = pd.concat([test_df, temp_df], axis=1)

    # Encode education and self_employed manually
    education_mapping = {'Graduate': 1, 'Not Graduate': 0}
    test_df['education'] = test_df['education'].map(education_mapping)
    self_employed_mapping = {'Yes': 1, 'No': 0}
    test_df['self_employed'] = test_df['self_employed'].map(self_employed_mapping)

    # Ensure the column order matches the order used during training
    final_columns = [
        'no_of_dependents', 'loan_amount', 'loan_term', 'cibil_score', 
        'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value', 
        'income_to_loan_ratio', 'loan_term_to_income_ratio', 
        'education', 'City_Ankara', 'City_Bursa', 'City_Erzurum', 
        'City_Istanbul', 'City_Izmir', 'self_employed'
    ]

    
    test_df = test_df[final_columns]
        

    result = Model.predict(test_df)[0]
    return result

# Streamlit app
st.title("Loan Approval Prediction")
st.header("Enter Loan Details")

# Collect user input
no_of_dependents = st.slider("Number of dependents", min_value=0, max_value=5, value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
City = st.selectbox("City", cities)
income_annum = st.slider("Annual Income", min_value=200000, max_value=9900000, value=200000, step=1000)
loan_amount = st.slider("Loan Amount", min_value=300000, max_value=39500000, value=300000, step=10000)
loan_term = st.slider("Loan Term", min_value=2, max_value=20, value=2, step=1)
cibil_score = st.slider("Credit Score", min_value=300, max_value=900, value=300, step=1)
residential_assets_value = st.slider("Residential AV", min_value=-100000, max_value=29100000, value=0, step=10000)
commercial_assets_value = st.slider("Commercial AV", min_value=0, max_value=19400000, value=0, step=1000)
luxury_assets_value = st.slider("Luxury AV", min_value=300000, max_value=39200000, value=300000, step=10000)
bank_asset_value = st.slider("Bank AV", min_value=0, max_value=14700000, value=0, step=10000)

# Predict loan approval
if st.button("Predict"):
    result = predict(no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score,
                     residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value, City)
    label = ["rejected", "approved"]
    st.markdown(f"## The application is **{label[result]}**.")
