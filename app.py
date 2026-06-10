from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import joblib
import sys
import easyocr
import re
import os
import tempfile

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
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    analysis_type: str = Form(...)
):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    content = await file.read()
    temp_file.write(content)
    temp_file.close()

    try:
        ocr_text = extract_text_from_image(temp_file.name)

        if analysis_type == "diabetes":
            result = predict_diabetes(ocr_text, diabetes_model)

        elif analysis_type == "both":
            celiac_result = predict_hybrid(ocr_text, model)
            diabetes_result = predict_diabetes(ocr_text, diabetes_model)

            result = {
                "sonuc": "BOTH",
                "celiac_sonuc": celiac_result["sonuc"],
                "celiac_aciklama": celiac_result["aciklama"],
                "celiac_bulunanlar": celiac_result["bulunanlar"],
                "diabetes_sonuc": diabetes_result["sonuc"],
                "diabetes_aciklama": diabetes_result["aciklama"],
                "diabetes_bulunanlar": diabetes_result["bulunanlar"],
            }

        else:
            result = predict_hybrid(ocr_text, model)

        return result

    finally:
        os.unlink(temp_file.name)

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

# Gluten iceren bilesenler (Ingilizce + Turkce)
# Kelime sinirlariyla aratacagiz, false-positive olmasin diye.
GLUTEN_KEYWORDS = {
    # ingilizce
    "wheat": "wheat",
    "barley": "barley",
    "rye": "rye",
    "malt": "malt",
    "semolina": "semolina",
    "durum": "durum bugday",
    "graham": "graham unu",
    "spelt": "spelt",
    "kamut": "kamut",
    "einkorn": "einkorn",
    "triticale": "triticale",
    "bulgur": "bulgur",
    "couscous": "couscous",
    "farro": "farro",
    "seitan": "seitan",
    "gluten": "gluten",
    # turkce
    "bugday": "bugday",
    "buğday": "bugday",
    "arpa": "arpa",
    "cavdar": "cavdar",
    "çavdar": "cavdar",
    "irmik": "irmik",
    "i̇rmik": "irmik",
    "kepek": "kepek (genellikle bugday kepegi)",
    "kuskus": "kuskus",
    "makarna unu": "makarna unu (bugday)",
}

def clean_text(text: str) -> str:
    """Metni kucult, ozel karakterleri temizle (model icin)."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tr_to_ascii(text: str) -> str:
    """Turkce karakterleri ASCII'ye cevir (sozluk eslemesi icin)."""
    tr_map = str.maketrans({
        "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g",
        "ı": "i", "İ": "i", "ö": "o", "Ö": "o",
        "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
        "â": "a", "î": "i", "û": "u",
    })
    return text.translate(tr_map)


def normalize_for_search(text: str) -> str:
    """Sozluk araması icin metni normalize et."""
    text = _tr_to_ascii(str(text)).lower()
    # noktalama -> bosluk
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return f" {text.strip()} "


def find_gluten_free_claim(text: str) -> bool:
    """Metinde 'glutensiz / gluten free' acik ifadesi var mi?"""
    norm = normalize_for_search(text)
    for phrase in GLUTEN_FREE_PHRASES:
        p_norm = normalize_for_search(phrase).strip()
        if f" {p_norm} " in norm:
            return True
    return False


def find_gluten_keywords(text: str):
    """Metinde gecen gluten anahtar kelimelerini dondur.
    ONCE 'gluten free / glutensiz' ifadelerini metinden cikarir,
    boylece 'GLUTEN FREE OATS' icinde 'gluten' kelimesi YANLIS yakalanmaz.
    """
    norm = normalize_for_search(text)
    # Glutensiz ifadelerini metinden cikar (kelime smearing'i onlemek icin bosluga cevir)
    for phrase in GLUTEN_FREE_PHRASES:
        p_norm = normalize_for_search(phrase).strip()
        norm = norm.replace(f" {p_norm} ", "  ")

    hits = []
    seen = set()
    for kw, label in GLUTEN_KEYWORDS.items():
        kw_norm = normalize_for_search(kw).strip()
        if not kw_norm:
            continue
        # Tam kelime eslemesi
        pattern = rf"(?<![a-z0-9]){re.escape(kw_norm)}(?![a-z0-9])"
        if re.search(pattern, norm):
            if label not in seen:
                seen.add(label)
                hits.append(label)
    return hits

# ============ DIYABET / SEKER ANALIZI ============

SUGAR_LOW = 5.0
SUGAR_HIGH = 22.5

SUGAR_FREE_PHRASES = [
    "sugar free", "sugar-free", "sugarfree",
    "no added sugar", "no sugar added", "without added sugar",
    "zero sugar", "0 sugar",
    "sekersiz", "şekersiz",
    "seker ilavesiz", "şeker ilavesiz",
    "seker icermez", "şeker içermez",
    "ilave seker icermez", "ilave şeker içermez",
    "dusuk sekerli", "düşük şekerli",
    "diyabetik", "diyabetiklere uygun",
    "diabetic", "diabetic friendly",
]

SUGAR_KEYWORDS = {
    "sugar": "sugar",
    "seker": "şeker",
    "şeker": "şeker",
    "glucose": "glikoz",
    "glikoz": "glikoz",
    "fructose": "fruktoz",
    "fruktoz": "fruktoz",
    "sucrose": "sükroz",
    "sukroz": "sükroz",
    "sükroz": "sükroz",
    "dextrose": "dekstroz",
    "dekstroz": "dekstroz",
    "maltose": "maltoz",
    "maltoz": "maltoz",
    "maltodextrin": "maltodekstrin",
    "maltodekstrin": "maltodekstrin",
    "corn syrup": "mısır şurubu",
    "misir surubu": "mısır şurubu",
    "mısır şurubu": "mısır şurubu",
    "high fructose corn syrup": "yüksek fruktozlu mısır şurubu (HFCS)",
    "hfcs": "yüksek fruktozlu mısır şurubu (HFCS)",
    "glucose syrup": "glikoz şurubu",
    "glikoz surubu": "glikoz şurubu",
    "glikoz şurubu": "glikoz şurubu",
    "nisasta surubu": "nişasta şurubu",
    "nişasta şurubu": "nişasta şurubu",
    "honey": "bal",
    "bal": "bal",
    "molasses": "pekmez/melas",
    "pekmez": "pekmez",
    "agave": "agave",
    "caramel": "karamel",
    "karamel": "karamel",
}

def find_sugar_free_claim(text: str) -> bool:
    norm = normalize_for_search(text)
    for phrase in SUGAR_FREE_PHRASES:
        p_norm = normalize_for_search(phrase).strip()
        if f" {p_norm} " in norm:
            return True
    return False


def find_sugar_keywords(text: str):
    norm = normalize_for_search(text)

    for phrase in SUGAR_FREE_PHRASES:
        p_norm = normalize_for_search(phrase).strip()
        norm = norm.replace(f" {p_norm} ", "  ")

    hits = []
    seen = set()

    for kw, label in SUGAR_KEYWORDS.items():
        kw_norm = normalize_for_search(kw).strip()
        pattern = rf"(?<![a-z0-9]){re.escape(kw_norm)}(?![a-z0-9])"

        if re.search(pattern, norm):
            if label not in seen:
                seen.add(label)
                hits.append(label)

    return hits


def extract_sugar_grams(text: str):
    norm = _tr_to_ascii(text).lower()

    patterns = [
        r"of\s*which\s*sugars?\s*[:\-]?\s*(\d+[.,]?\d*)\s*g",
        r"bunlardan\s*sekerler?\s*[:\-]?\s*(\d+[.,]?\d*)\s*g",
        r"sekerler?\s*[:\-]?\s*(\d+[.,]?\d*)\s*g",
        r"sugars?\s*[:\-]?\s*(\d+[.,]?\d*)\s*g",
    ]

    for pat in patterns:
        m = re.search(pat, norm)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                pass

    return None

def predict_diabetes(ocr_text: str, model):
    clean_input = clean_text(ocr_text)

    if not clean_input:
        return {
            "sonuc": "BILINMIYOR",
            "aciklama": "Gorselden okunabilir yazi tespit edilemedi.",
            "bulunanlar": []
        }

    sugar_g = extract_sugar_grams(ocr_text)
    is_sf_claim = find_sugar_free_claim(ocr_text)
    sugar_hits = find_sugar_keywords(ocr_text)

    model_pred = int(model.predict([clean_input])[0])
    model_proba = model.predict_proba([clean_input])[0]

    if sugar_g is not None:
        if sugar_g < SUGAR_LOW:
            seviye = "DUSUK"
        elif sugar_g <= SUGAR_HIGH:
            seviye = "ORTA"
        else:
            seviye = "YUKSEK"

        aciklama = f"Besin degerleri tablosundan 100 g basina yaklasik {sugar_g:g} g seker tespit edildi."

    elif is_sf_claim and not sugar_hits:
        seviye = "DUSUK"
        aciklama = "Etikette sekersiz / sugar free ifadesi var ve ilave seker kaynagi gorulmedi."

    elif sugar_hits:
        if len(sugar_hits) >= 2 or model_pred == 1:
            seviye = "YUKSEK"
        else:
            seviye = "ORTA"

        aciklama = f"Tespit edilen seker kaynaklari: {', '.join(sugar_hits)}"

    else:
        if model_pred == 1:
            seviye = "ORTA"
            aciklama = "Anahtar kelime bulunamadi ancak model yuksek sekerli urun kalibi tespit etti."
        else:
            seviye = "DUSUK"
            aciklama = "Belirgin seker kaynagi tespit edilmedi, model yuksek risk gormuyor."

    return {
        "sonuc": seviye,
        "aciklama": aciklama,
        "bulunanlar": sugar_hits
    }

# ============ HIBRIT KARAR ============
def predict_hybrid(ocr_text: str, model) -> None:
    """Once kural, sonra modeli birlestiren hibrit karar."""
    clean_input = clean_text(ocr_text)
    if not clean_input:
        print("UYARI: Gorselden hicbir yazi okunamadi. Daha net bir fotograf dene.")
        return

    # 1) Kural: glutensiz iddiasi var mi?
    is_gf_claim = find_gluten_free_claim(ocr_text)

    # 2) Kural: gluten kelimesi var mi?
    keyword_hits = find_gluten_keywords(ocr_text)
    rule_says_risky = len(keyword_hits) > 0

    # 3) Model tahmini
    model_pred = int(model.predict([clean_input])[0])
    model_proba = model.predict_proba([clean_input])[0]

    # ----- HIBRIT KARAR MANTIGI -----
    # A) Acik "glutensiz" ifadesi varsa: model ne derse desin RISK YOK
    #    (ANCAK: ayni etiketin baska yerinde "BUGDAY UNU" gibi acik bir
    #     gluten kaynagi geciyorsa karari KARARSIZ yapariz)
    if is_gf_claim and not rule_says_risky:
        final_label = 0
        reason = "Etiket 'glutensiz / gluten free' diyor ve gluten kaynagi gorulmedi."
    elif is_gf_claim and rule_says_risky:
        # Celiski: "glutensiz" yaziyor ama icindekilerde bugday/arpa vb. var
        final_label = 1
        reason = (
            "Etikette 'glutensiz' yazsa da icindekilerde su gluten kaynaklari tespit "
            f"edildi: {', '.join(keyword_hits)}. (OCR hatasi olabilir; etiketi kontrol et.)"
        )
    elif rule_says_risky:
        final_label = 1
        reason = f"Su gluten kaynaklari tespit edildi: {', '.join(keyword_hits)}"
    else:
        # Kural net bir sey demiyor: modele guven
        final_label = model_pred
        if model_pred == 1:
            reason = "Kural sozluğu net bir gluten kaynagi bulamadi ama model riskli buluyor (su¨pheli kalip)."
        else:
            reason = "Hicbir gluten anahtar kelimesi bulunamadi, model risk gormuyor."

    # Olasilik gosterimi
    if final_label == 1:
        risk_pct = max(model_proba[1] * 100, 95.0 if rule_says_risky else 0.0)
    else:
        risk_pct = (1 - model_proba[0]) * 100  # bilgi amacli

    # ----- CIKTI -----
    print("\n========== SONUC (COLYAK / GLUTEN) ==========")
    if final_label == 1:
        print("⚠️  COLYAK HASTALARI ICIN RISKLI  (gluten tespit edildi)")
    else:
        print("✅ COLYAK HASTALARI ICIN GUVENLI gozukuyor  (gluten tespit edilmedi)")
    print(f"\nAciklama : {reason}")
    print(f"Model tahmini   : {'RISKLI' if model_pred==1 else 'GUVENLI'} "
          f"(risk %{model_proba[1]*100:.1f} / guven %{model_proba[0]*100:.1f})")
    print(f"Kural tespiti   : "
          f"{'gluten kaynagi VAR -> ' + ', '.join(keyword_hits) if rule_says_risky else 'gluten kaynagi yok'}"
          f"{'  |  GLUTENSIZ ibaresi var' if is_gf_claim else ''}")
    print("=============================================")
    print("\nNot: Bu tahmin bir on-degerlendirmedir. Colyak hastalari icin kesin")
    print("karar etiketin TAM ve OKUNAKLI hali ile, gerekirse uretici/diyetisyene")
    print("danisarak verilmelidir. OCR hatalari yanlis sonuca yol acabilir.")
    return {
    "sonuc": "RISKLI" if final_label == 1 else "GUVENLI",
    "aciklama": reason,
    "bulunanlar": keyword_hits
}