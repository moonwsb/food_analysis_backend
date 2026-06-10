from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("colyak_model.pkl")
diabetes_model = joblib.load("diyabet_model.pkl")

@app.get("/")
def home():
    return {"message": "API calisiyor"}