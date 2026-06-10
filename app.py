from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("colyak_model.pkl")
diabetes_model = joblib.load("diyabet_model.pkl")

@app.get("/")
def home():
    return {"message": "API calisiyor"}

@app.get("/test")
def test():
    return {
        "celiac_model": "loaded",
        "diabetes_model": "loaded"
    }
@app.get("/predict")
def predict_test():
    return {"status": "working"}