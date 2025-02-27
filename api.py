
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("inflation_model.pkl")
print('model: ', model)
scaler = joblib.load("scaler.pkl")
print(scaler.feature_names_in_)  

@app.post("/predict/")
def predicted_inflation(data: dict):
    df = pd.DataFrame([data])
    df = df.astype(float)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)
    
    return {"predicted_inflation_rate":prediction.tolist()}