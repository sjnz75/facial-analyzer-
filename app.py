# app.py — Facial Aesthetic Analyzer **v1.1** (interattivo e stabile)
# ==================================================================
# Autore: ChatGPT (OpenAI)
# Licenza: MIT
"""
Dipendenze (requirements.txt) – usa **esattamente** queste versioni:

    streamlit==1.34.0
    mediapipe==0.10.21
    opencv-python-headless==4.7.0.72   # solo versione headless, niente libGL
    numpy
    pillow==9.5.0                      # compatibile col canvas
    streamlit-drawable-canvas==0.9.3   # canvas interattivo

> Dopo aver aggiornato `requirements.txt`, fai **Clear cache & reboot** in Streamlit Cloud, poi Deploy.
"""

import streamlit as st
import cv2, mediapipe as mp, numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ─────────────────── Streamlit page config ────────────────────
st.set_page_config(page_title="Facial Analyzer v1.1", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v1.1 (interattivo)")
st.markdown("""
**Istruzioni**  
1. Carica una foto frontale in *Natural Head Position*.  
2. Le linee di riferimento compaiono automaticamente.  
3. **Trascina** le linee per posizionarle con precisione.  
4. Premi **Ricalcola diagnosi** per aggiornare proporzioni e report.
""")

# ─────────────────── Costanti e modelli ───────────────────────
MP_FACE = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                          max_num_faces=1,
                                          refine_landmarks=True)
LM = {
    "glabella": 9,
    "subnasale": 2,
    "pogonion": 152,
    "eye_L": 33,
    "eye_R": 263,
    "mouth_L": 61,
    "mouth_R": 291,
    "ala_L": 98,
    "ala_R": 327,
    "upper_incisor": 13,
}
PAIRS_SYMM = [
    (LM["eye_L"], LM["eye_R"]),
    (LM["mouth_L"], LM["mouth_R"]),
    (LM["ala_L"], LM["ala_R"]),
]

# Range accettabili
RANGE = {
    "terzo": "33 ± 3 %",
    "midline": "< 0.4 % larghezza",
    "simmetria": "< 3 %",
}

# ─────────────────── Upload immagine ──────────────────────────
file = st.file_uploader("🖼️ Carica immagine JPG/PNG", type=["jpg", "jpeg", "png"])
if not file:
    st.stop()

img_pil = Image.open(file).convert("RGB")
img_pil = ImageOps.exif_transpose(img_pil)
img = np.array(img_pil)
H, W = img.shape[:2]

# ─────────────────── Landmark detection ───────────────────────
res = MP_FACE.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato. Controlla illuminazione e posizione.")
    st.stop()

lm = res.multi_face_landmarks[0].landmark
P = lambda i: np.array([lm[i].x * W, lm[i].y * H])

# ─────────────────── Linee iniziali ───────────────────────────
y_glab, y_subn, y_pogo = P(LM["glabella"])[1], P(LM["subnasale"])[1], P(LM["pogonion"])[1]
y_trich = max(0, int(2 * y_glab - y_subn))            # stima automatica
y_bip = int((P(LM["eye_L"])[1] + P(LM["eye_R"])[1]) / 2)

init_lines = {
    "Trichion":       ((0, y_trich), (W, y_trich), "#1E90FF"),  # DodgerBlue
    "Glabella":       ((0, int(y_glab)), (W, int(y_glab)), "#1E90FF"),
    "Bipupillare":    ((0, y_bip), (W, y_bip), "#1E90FF"),
    "Subnasale":      ((0, int(y_subn)), (W, int(y_subn)), "#1E90FF"),
    "Interalare":     ((0, int(P(LM["ala_L"])[1])), (W, int(P(LM["ala_L"])[1])), "#1E90FF"),
    "Menton":         ((0, int(y_pogo)), (W, int(y_pogo)), "#1E90FF"),
    "Midline":        ((W // 2, 0), (W // 2, H), "#32CD32"),   # LimeGreen
    "Interincisale":  ((int(P(LM["upper_incisor"])[0]), 0), (int(P(LM["upper_incisor"])[0]), H), "#FF1493"),  # DeepPink
}

init_json = {"version": "4.4.0", "objects": []}
for name, ((x1, y1), (x2, y2), color) in init_lines.items():
    init_json["objects"].append({
        "type": "line",
        "name": name,
        "stroke": color,
        "strokeWidth": 2,
        "x1": x1, "y1": y1,
        "x2": x2, "y2": y2,
    })

st.write("### 1️⃣ Sposta le linee se necessario, poi premi **Ricalcola diagnosi**")
canvas = st_canvas(
    background_image=img_pil,
    initial_drawing=init_json,
    update_streamlit=True,
    height=H,
    width=W,
    drawing_mode="transform",      # trascina
    key="canvas",
)

# Helper per recuperare X/Y dal canvas
objs = {
    o["name"]: o for o in canvas.json_data["objects"] if o["type"] == "line"
} if canvas.json_data else {}

def line_center(obj, axis="y"):
    if axis == "y":
        return float(obj["y1"] + obj["y2"]) / 2.0
    return float(obj["x1"] + obj["x2"]) / 2.0

if st.button("Ricalcola diagnosi"):
    # ─────── Coordinate finali (dopo drag) ───────
    try:
        y_T = line_center(objs["Trichion"], "y")
        y_G = line_center(objs["Glabella"], "y")
        y_S = line_center(objs["Subnasale"], "y")
        y_M = line_center(objs["Menton"], "y")
        x_mid = line_center(objs["Midline"], "x")
        x_inc = line_center(objs["Interincisale"], "x")
    except KeyError:
        st.error("Alcune linee fondamentali sono state cancellate: ricarica la pagina.")
        st.stop()

    # ─────── Metriche percentuali ───────
    H_tot = y_M - y_T
    thirds_pct = np.array([
        (y_G - y_T) / H_tot * 100,
        (y_S - y_G) / H_tot * 100,
        (y_M - y_S) / H_tot * 100,
    ])

    off_pct = abs(x_inc - x_mid) / W * 100
    symm_pct = np.sqrt(np.mean([
        abs(P(a)[0] - (W - P(b)[0])) ** 2 for a, b in PAIRS_SYMM
    ])) / W * 100

    # ─────── UI – Tabella risultati ───────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Proporzioni facciali (%)")
        st.json({"Superiore": round(float(thirds_pct[0]), 1),
                 "Medio": round(float(thirds_pct[1]), 1),
                 "Inferiore": round(float(thirds_pct[2]), 1)})
    with col2:
        st.subheader("📐 Deviazioni (%)")
        st.json({"Offset midline": round(float(off_pct), 2),
                 "Simmetria": round(float(symm_pct), 2)})

    # ─────── Diagnosi ───────
    st.subheader("📝 Diagnosi" )
    diagnosis = []

    # Proporzioni
    if np.all(np.abs(thirds_pct - 33) < 3):
        diagnosis.append("Proporzioni: nella norma (33 ± 3 %).")
    else:
        if thirds_pct[2] > ((thirds_pct[0] + thirds_pct[1]) / 2) * 1.1:
            diagnosis.append("Long face: terzo inferiore >110 % della media superiore+media.")
        elif thirds_pct[2] < ((thirds_pct[0] + thirds_pct[1]) / 2) * 0.9:
            diagnosis.append("Short face: terzo inferiore <90 % della media superiore+media.")
        else:
            diagnosis.append("Proporzioni: fuori dal range ideale 33 ± 3 %." )

    # Midline
    diagnosis.append("Midline: ok" if off_pct < 0.4 else "Midline: deviata (>0.4 %).")

    # Simmetria
    diagnosis.append("Simmetria: entro 3 %." if symm_pct < 3 else "Asimmetria: >3 %.")

    st.markdown("\n".join(["- " + d for d in diagnosis]))

    # ─────── Range accettabili ───────
    st.subheader("🏷️ Range clinici di riferimento")
    st.json(RANGE)

    st.caption("Valori in percentuale rispetto alle dimensioni facciali; usare come supporto clinico, non sostitutivo di esame diretto.")
