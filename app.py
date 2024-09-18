from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import logging


logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s : %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    Model = joblib.load("loan_approval_model.pkl")
    Scaler = joblib.load("rb_scaler.pkl")
    logger.info("Model and Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise e


cities = ['Ankara', 'Bursa', 'Erzurum', 'Istanbul', 'Izmir']

# Feature engineering fonksiyonu
def feature_engineering(df):
    try:
        df['income_to_loan_ratio'] = df['loan_amount'] / df['income_annum']
        df['loan_term_to_income_ratio'] = df['loan_term'] / df['income_annum']
        df['loan_term'] = df['loan_term'] * 36
        df.drop(columns=['income_annum'], inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise e

# Şehir kodlama fonksiyonu
def encode_city(df, city):
    city_encoder = OneHotEncoder(sparse_output=False, categories=[cities])
    city_encoded = city_encoder.fit_transform(df[['City']])
    city_encoded_df = pd.DataFrame(city_encoded, columns=[f"City_{c}" for c in cities])
    df = pd.concat([df, city_encoded_df], axis=1)
    df.drop(columns=['City'], inplace=True)
    return df

# Tahmin fonksiyonu
def predict_loan(data):
    df = pd.DataFrame([data])

    # Feature engineering işlemleri
    df = feature_engineering(df)

    # Şehir kodlama
    df = encode_city(df, df['City'].values[0])

    # Sayısal değerleri ölçeklendirme
    scaled_numerical_cols = ['loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                             'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
    df[scaled_numerical_cols] = Scaler.transform(df[scaled_numerical_cols])

    # education ve self_employed sütunlarını sayısal hale getirme
    education_mapping = {'Graduate': 1, 'Not Graduate': 0}
    df['education'] = df['education'].map(education_mapping)
    
    self_employed_mapping = {'Yes': 1, 'No': 0}
    df['self_employed'] = df['self_employed'].map(self_employed_mapping)

    # Sütun sırasını modele uygun hale getir
    final_columns = [
        'no_of_dependents', 'loan_amount', 'loan_term', 'cibil_score',
        'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value',
        'income_to_loan_ratio', 'loan_term_to_income_ratio',
        'education', 'City_Ankara', 'City_Bursa', 'City_Erzurum',
        'City_Istanbul', 'City_Izmir', 'self_employed'
    ]
    df = df[final_columns]

    # Modelden tahmin al
    prediction = Model.predict(df)[0]
    return prediction

# Ana sayfa: Giriş formunu görüntüle
@app.route('/')
def home():
    return render_template('index.html')

# Tahmin sonucunu döner
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Formdan gelen verileri al ve tip dönüşümü yap
        data = {
            "no_of_dependents": int(request.form['no_of_dependents']),
            "education": request.form['education'],
            "self_employed": request.form['self_employed'],
            "income_annum": float(request.form['income_annum']),
            "loan_amount": float(request.form['loan_amount']),
            "loan_term": float(request.form['loan_term']),
            "cibil_score": float(request.form['cibil_score']),
            "residential_assets_value": float(request.form['residential_assets_value']),
            "commercial_assets_value": float(request.form['commercial_assets_value']),
            "luxury_assets_value": float(request.form['luxury_assets_value']),
            "bank_asset_value": float(request.form['bank_asset_value']),
            "City": request.form['City']
        }
        
        # Model üzerinden tahmin al
        prediction = predict_loan(data)
        result = "approved" if prediction == 1 else "rejected"

        return render_template('result.html', prediction_text=f"Loan Status: {result}")
    
    except Exception as e:
        logger.error(f"Error in prediction request: {str(e)}")
        return render_template('result.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
