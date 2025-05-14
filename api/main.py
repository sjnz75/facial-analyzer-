from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
from utils.geometry import line_angle_deg, line_deviation_mm, segment_length, compute_vertical_thirds
from utils.diagnostics import diagnose_median_line, diagnose_angle_line, diagnose_thirds

app = FastAPI(title="Face Aesthetics Analyzer API")

class Landmarks(BaseModel):
    points: dict  # es. {"Gl": [x,y], ...}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), lm: Landmarks = None):
    # Leggi immagine
    img = Image.open(file.file).convert("RGB")
    if not lm or len(lm.points) < 11:
        raise HTTPException(400, "Landmarks insufficienti.")
    pts = lm.points
    # Esegui stessi calcoli di app.py
    med_angle = line_angle_deg(tuple(pts["Gl"]), tuple(pts["Pg"]))
    med_dev = line_deviation_mm(tuple(pts["Gl"]), tuple(pts["Sn"]), tuple(pts["Pg"]))
    ip_angle = line_angle_deg(tuple(pts["PR"]), tuple(pts["PL"]), horizontal=True)
    com_angle = line_angle_deg(tuple(pts["CR"]), tuple(pts["CL"]), horizontal=True)
    ala_diff = abs(pts["AR"][1] - pts["AL"][1])
    thirds = compute_vertical_thirds(tuple(pts["Tri"]), tuple(pts["Gl"]), tuple(pts["Sn"]), tuple(pts["Me"]))
    # Diagnostica
    diag = {
        "median": diagnose_median_line(med_dev),
        "interpupillare": diagnose_angle_line(ip_angle, "interpupillare"),
        "commissurale": diagnose_angle_line(com_angle, "commissurale"),
        "interalare": "Asimmetria" if ala_diff>2 else "Normale",
        "terzi": diagnose_thirds(thirds)
    }
    return {
        "measures": {
            "median_deviation_mm": med_dev,
            "median_angle_deg": med_angle,
            "interpupillare_angle_deg": ip_angle,
            "commissurale_angle_deg": com_angle,
            "interalare_diff_mm": ala_diff,
            "vertical_thirds_pct": thirds
        },
        "diagnosis": diag
    }
