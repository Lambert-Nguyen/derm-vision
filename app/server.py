#!/usr/bin/env python3
"""
DermVision FastAPI server.
Serves the web app (app/web/) and provides model inference API.

Run: python app/server.py
  or: uvicorn app.server:app --reload --port 8000
"""
import os, sys, io, random, base64
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.dataset import CLASS_NAMES
from src.models.efficientnet import EfficientNetB3Classifier
from src.transforms import get_val_transforms
from src.gradcam import generate_gradcam

# ── Globals ──────────────────────────────────────────────────────────────────
MODEL = None
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
IMAGE_SIZE = 300
WEB_DIR = Path(__file__).parent / "web"
CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "checkpoints"
    / "efficientnet-b3_cardassian-spot-5" / "best_model-2.pth"
)
SAMPLES_DIR = PROJECT_ROOT / "data" / "raw" / "ISIC_2019_Training_Input"

CLASS_INFO = {
    "MEL":   {"name": "Melanoma",                    "risk": "high",
              "desc": "A serious form of skin cancer that develops in melanocytes."},
    "NV":    {"name": "Melanocytic Nevus",            "risk": "low",
              "desc": "A common benign mole formed by a cluster of melanocytes."},
    "BCC":   {"name": "Basal Cell Carcinoma",         "risk": "high",
              "desc": "The most common skin cancer. Slow-growing, rarely spreads if treated early."},
    "AKIEC": {"name": "Actinic Keratosis / Bowen's",  "risk": "moderate",
              "desc": "A pre-cancerous rough skin patch caused by years of sun exposure."},
    "BKL":   {"name": "Benign Keratosis",             "risk": "low",
              "desc": "Non-cancerous skin growth including seborrheic keratosis and solar lentigo."},
    "DF":    {"name": "Dermatofibroma",               "risk": "low",
              "desc": "A common benign fibrous nodule typically found on the lower legs."},
    "VASC":  {"name": "Vascular Lesion",              "risk": "low",
              "desc": "Abnormality of blood vessels including cherry angioma and pyogenic granuloma."},
    "SCC":   {"name": "Squamous Cell Carcinoma",      "risk": "high",
              "desc": "The second most common skin cancer. Can spread if left untreated."},
}

RISK_STYLES = {
    "high":     {"label": "High Risk",     "color": "#e24b4a", "bg": "#fcebeb", "text": "#501313"},
    "moderate": {"label": "Moderate Risk", "color": "#ef9f27", "bg": "#faeeda", "text": "#633806"},
    "low":      {"label": "Low Risk",      "color": "#1d9e75", "bg": "#e1f5ee", "text": "#085041"},
}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    if CHECKPOINT.exists():
        try:
            MODEL = EfficientNetB3Classifier(num_classes=8, pretrained=False)
            MODEL.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
            MODEL.to(DEVICE)
            MODEL.eval()
            print(f"✓ Model loaded  [{DEVICE}]  {CHECKPOINT.name}")
        except Exception as e:
            print(f"⚠ Model load failed: {e} — demo mode")
    else:
        print(f"⚠ No checkpoint at {CHECKPOINT} — demo mode")
    yield


app = FastAPI(title="DermVision", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Page routes ───────────────────────────────────────────────────────────────
@app.get("/")
async def homepage():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/analyze")
async def analyze_page():
    return FileResponse(WEB_DIR / "analyze.html")

@app.get("/about")
async def about_page():
    return FileResponse(WEB_DIR / "about.html")


# ── API ───────────────────────────────────────────────────────────────────────
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    if MODEL is None:
        probs = {name: 1.0 / 8 for name in CLASS_NAMES}
    else:
        img_np = np.array(pil_image)
        t = get_val_transforms(IMAGE_SIZE)
        tensor = t(image=img_np)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p = torch.softmax(MODEL(tensor), dim=1).cpu().numpy()[0]
        probs = {name: float(v) for name, v in zip(CLASS_NAMES, p)}

    sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_cls, top_prob = sorted_p[0]
    info = CLASS_INFO[top_cls]
    risk = RISK_STYLES[info["risk"]]

    return JSONResponse({
        "predictions":        probs,
        "sorted_predictions": [{"class": c, "name": CLASS_INFO[c]["name"],
                                 "probability": round(p, 4),
                                 "risk": CLASS_INFO[c]["risk"]}
                                for c, p in sorted_p],
        "top_class":          top_cls,
        "top_name":           info["name"],
        "top_probability":    round(top_prob, 4),
        "risk_label":         risk["label"],
        "risk_color":         risk["color"],
        "risk_bg":            risk["bg"],
        "risk_text":          risk["text"],
        "description":        info["desc"],
        "advice":             ("Please consult a dermatologist as soon as possible."
                               if info["risk"] == "high" else
                               "Schedule a dermatologist appointment for evaluation."
                               if info["risk"] == "moderate" else
                               "Appears likely benign. Continue monitoring for changes."),
        "demo_mode":          MODEL is None,
    })


@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...), target_class: int = -1):
    """Return a Grad-CAM heatmap overlay (PNG) for the uploaded image.

    target_class: optional class index (0-7). -1 → use predicted class.
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded — Grad-CAM unavailable in demo mode")

    content = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    target = target_class if target_class >= 0 else None
    target_layer = MODEL.backbone.conv_head

    try:
        overlay = generate_gradcam(
            model=MODEL,
            image_path=pil_image,
            target_layer=target_layer,
            image_size=IMAGE_SIZE,
            target_class=target,
            device=DEVICE,
        )
    except Exception as e:
        raise HTTPException(500, f"Grad-CAM failed: {e}")
    finally:
        # generate_gradcam flips requires_grad=True on every param; restore eval state
        for p in MODEL.parameters():
            p.requires_grad = False

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return JSONResponse({"image": f"data:image/png;base64,{b64}"})


@app.get("/api/samples")
async def get_samples():
    if not SAMPLES_DIR.exists():
        return JSONResponse({"available": False, "images": []})
    images = list(SAMPLES_DIR.glob("*.jpg"))
    if not images:
        return JSONResponse({"available": False, "images": []})
    sample = random.sample(images, min(20, len(images)))
    return JSONResponse({
        "available": True,
        "images": [f"/images/{img.name}" for img in sample],
    })


# ── Static files (must be last) ───────────────────────────────────────────────
if SAMPLES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(SAMPLES_DIR)), name="images")

app.mount("/", StaticFiles(directory=str(WEB_DIR)), name="static")


if __name__ == "__main__":
    print("\n  DermVision")
    print(f"  ─────────────────────────────")
    print(f"  Homepage  →  http://localhost:8000")
    print(f"  Analysis  →  http://localhost:8000/analyze")
    print(f"  API docs  →  http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
