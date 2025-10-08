from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import pickle
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

import io
from fastapi.responses import Response, StreamingResponse

app = FastAPI(title="Fraud Detection API")
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:5001",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5001",
    "https://finguard-frontend.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # during dev you can use ["*"], but restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 0️⃣ Load calibrated XGBoost + threshold
# -------------------------------
with open("fraud_model/xgb_calibrator.pkl", "rb") as f:
    data = pickle.load(f)

# Load encoders
with open("fraud_model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


calibrator = data["calibrator"]   # CalibratedClassifierCV object
threshold = data["threshold"]     # float threshold to decide fraud
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unnecessary columns
    for col in ['Unnamed: 0', 'cc_num', 'trans_num']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Extract date features
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day'] = df['trans_date_trans_time'].dt.day
        df['weekday'] = df['trans_date_trans_time'].dt.weekday
        df['month'] = df['trans_date_trans_time'].dt.month
        df = df.drop(columns=['trans_date_trans_time'])

    # Encode categorical columns using saved LabelEncoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Convert 'dob' to numeric (YYYYMMDD)
    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce').dt.strftime('%Y%m%d').astype(float)

    # Convert other object columns if any
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).map(lambda x: -1)  # fallback

    return df


# -------------------------------
# 1️⃣ Manual transaction prediction
# -------------------------------
@app.post("/predict/manual")
def predict_manual(transaction: dict):
    try:
        df = pd.DataFrame([transaction])
        df = preprocess(df)

        # 🔧 Enforce column order
        EXPECTED_COLUMNS = [
            'merchant', 'category', 'amt', 'first', 'last', 'gender', 'street',
            'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob',
            'unix_time', 'merch_lat', 'merch_long', 'hour', 'day', 'month', 'weekday'
        ]
        df = df.reindex(columns=EXPECTED_COLUMNS)

        # Calibrated probability
        probability = calibrator.predict_proba(df)[:, 1][0]
        prediction = int(probability > threshold)

        return {"is_fraud": prediction, "fraud_probability": float(probability)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------
# 2️⃣ CSV upload batch prediction
# -------------------------------
@app.post("/predict/csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df_processed = preprocess(df)

        # 🧩 Extract expected column order from model if possible
        try:
            expected_columns = calibrator.base_estimator_.get_booster().feature_names
        except Exception:
            # fallback to static order if not found
            expected_columns = [
                'merchant', 'category', 'amt', 'first', 'last', 'gender', 'street',
                'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob',
                'unix_time', 'merch_lat', 'merch_long', 'hour', 'day', 'month', 'weekday'
            ]

        # 🧱 Align dataframe with model’s expected order
        df_processed = df_processed.reindex(columns=expected_columns)

        # 🧮 Predictions
        probabilities = calibrator.predict_proba(df_processed)[:, 1]
        predictions = (probabilities > threshold).astype(int)

        df['is_fraud'] = predictions
        df['fraud_probability'] = probabilities

        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sample-csv")
def sample_csv():
    # Base sample data (without is_fraud)
    sample_content = (
        "trans_date_trans_time,cc_num,merchant,category,amt,first,last,gender,street,city,state,zip,lat,long,city_pop,job,dob,trans_num,unix_time,merch_lat,merch_long\n"
        "2019-01-01 00:00:18,2703186189652095,fraud_Ruecker Group,misc_net,843.91,Jason,Murphy,M,542 Steve Curve Suite 011,Collettsville,NC,28611,35.9946,-81.7266,885,Soil scientist,1988-09-15,2f7d497f607396ab669c14c2abe3886f,1325548328,35.9856,-81.3833\n"
        "2019-01-03 01:05:27,340187018810220,fraud_Conroy-Cruickshank,gas_transport,10.76,Misty,Hart,F,27954 Hall Mill Suite 575,San Antonio,TX,78208,29.44,-98.459,1595797,Horticultural consultant,1960-10-28,0a2f8002e55a3565c5c88d8cf039fed8,1325552727,28.8567,-97.7942\n"
    )
    df = pd.read_csv(io.StringIO(sample_content))

    # Use the same preprocessing function as /predict/manual and /predict/csv
    df_processed = preprocess(df)

    # Ensure columns are in the same order as model expects
    model_cols = calibrator.base_estimator.feature_names_in_
    df_processed = df_processed[model_cols]

    # Predict
    probabilities = calibrator.predict_proba(df_processed)[:, 1]
    predictions = (probabilities > threshold).astype(int)

    # Add predictions to CSV
    df['is_fraud'] = predictions
    df['fraud_probability'] = probabilities

    # Return CSV
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    return Response(
        content=stream.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample.csv"}
    )

