from fastapi import FastAPI
from PIL import Image
import joblib
import sys
import easyocr
import re

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