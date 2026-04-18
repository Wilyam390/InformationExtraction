"""
api/main.py
===========
FastAPI backend for DocClassify React UI.

Runs on http://localhost:8000
Endpoints:
  POST /api/classify       — JSON body { "text": "..." }
  POST /api/classify-file  — multipart/form-data with 'file' field
  GET  /api/health         — health check

Launch (from repo root):
    py -m uvicorn api.main:app --reload
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict, _load_models

_load_models()

app = FastAPI(title="DocClassify API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/classify")
def classify_text(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    result = predict(req.text)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/api/classify-file")
async def classify_file(file: UploadFile = File(...)):
    allowed = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    result = predict(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result
