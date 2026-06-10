from fastapi import FastAPI
from PIL import Image
import joblib
import sys
import easyocr

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
@app.get("/ocr-test")
def ocr_test():

    return {"easyocr": "import edildi"}

def extract_text_from_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as im:
            im.verify()
    except Exception as e:
        sys.exit(
            f"HATA: '{image_path}' acilabilir bir gorsel degil ({e}).\n"
            "Lutfen jpg/png gibi bir FOTOGRAF sec, CSV/Excel degil."
        )

   
    print(f"[OCR] Gorsel okunuyor: {image_path}")
    reader = easyocr.Reader(["en", "tr"])
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return " ".join(result)