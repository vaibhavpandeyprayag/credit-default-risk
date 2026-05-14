from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware


# Load the model
model = joblib.load("models/random_forest_model.joblib")

app = FastAPI(title="Credit Default Risk Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://creditdefaultrisk.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema
class CandidateData(BaseModel):
    duration_months: int
    credit_amount: int
    age: int
    
    checking_account_status: str
    credit_history: str
    savings_account: str
    
    purpose: str
    property: str
    housing: str
    employment_since: str


@app.get("/")
def home():
    return {"message": "Credit Default Risk Prediction API Running"}


@app.post("/predict")
def predict(candidate_data: CandidateData):
    input_df = pd.DataFrame([candidate_data.dict()])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

# Risk label
    if probability < 0.3:
        risk_level = "Low Risk"
    elif probability < 0.6:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "decision": "Likely to Default" if prediction == 1 else "Likely Non-Defaulter"
    }