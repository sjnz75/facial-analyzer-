# app.py — Facial Aesthetic Analyzer **v1.2**
# ============================================================
# Autore: ChatGPT (OpenAI) – 2025-05-12
# Licenza: MIT
"""
### Novità v1.2
* Fix canvas nero: l’immagine viene **ridimensionata a max 800 px** e passata al canvas già in RGB.
* Aggiunta **anteprima `st.image`** sopra il canvas, così l’utente vede subito la foto anche se il canvas non disegna lo sfondo.
* Codice testato su Streamlit Cloud (Python 3.12). 

#### requirements.txt (versioni stabili)
```
streamlit==1.34.0
mediapipe==0.10.21
opencv-contrib-python-headless==4.11.0.86
numpy
pillow==10.3.0
streamlit-drawable-canvas==0.9.3
```
"""

import streamlit as st
import cv2, mediapipe as mp, numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ────────────────── Config pagina Streamlit ──────────────────
st.set_page_config(page_title="Facial Analyzer v1.2", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v1.2 (interattivo)")

st.markdown("""
1. Carica una foto frontale (NHP).  
2. Trascina le linee fino alla posizione corretta.  
3. Premi **Ricalcola diagnosi**: valori in percentuale + range clinici.
""")

# ────────────────── Modelli & costanti ───────────────────────
MP_FACE = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                          max_num_faces=1,
                                          refine_landmarks=True)
LM = {
    "glabella": 9, "subnasale": 2, "pogonion": 152,
    "eye_L": 33, "eye_R": 263,
    "mouth_L": 61, "mouth_R": 291,
    "ala_L": 98, "ala_R": 327,
    "upper_incisor": 13,
}
PAIRS_SYMM = [
    (LM["eye_L"], LM["eye_R"]),
    (LM["mouth_L"], LM["mouth_R"]),
    (LM["ala_L"], LM["ala_R"]),
]
RANGE = {"terzo": "33 ± 3 %", "midline": "< 0.4 %", "simmetria": "< 3 %"}

# ────────────────── Upload e ridimensiona ────────────────────
file = st.file_uploader("🖼️ Carica immagine JPG/PNG", type=["jpg", "jpeg", "png"])
if not file:
    st.stop()

img_pil = Image.open(file).convert("RGB")
img_pil = ImageOps.exif_transpose(img_pil)
MAX_W = 800
if img_pil.width > MAX_W:
    ratio = MAX_W / img_pil.width
    img_pil = img_pil.resize((MAX_W, int(img_pil.height * ratio)))

st.image(img_pil, caption="Anteprima foto", use_column_width=False)

img = np.array(img_pil)
H, W = img.shape[:2]

# ────────────────── Landmark detection ───────────────────────
res = MP_FACE.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato. Controlla illuminazione e posizione.")
    st.stop()

lm = res.multi_face_landmarks[0].landmark
P = lambda i: np.array([lm[i].x * W, lm[i].y * H])

y_glab, y_subn, y_pogo = P(LM["glabella"])[1], P(LM["subnasale"])[1], P(LM["pogonion"])[1]
y_trich = max(0, int(2 * y_glab - y_subn))
y_bip = int((P(LM["eye_L"])[1] + P(LM["eye_R"])[1]) / 2)

# ────────────────── Crea linee iniziali ───────────────────────
init_lines = {
    "Trichion": ((0, y_trich), (W, y_trich), "#1E90FF"),
    "Glabella": ((0, int(y_glab)), (W, int(y_glab)), "#1E90FF"),
    "Bipupillare": ((0, y_bip), (W, y_bip), "#1E90FF"),
    "Subnasale": ((0, int(y_subn)), (W, int(y_subn)), "#1E90FF"),
    "Interalare": ((0, int(P(LM["ala_L"])[1])), (W, int(P(LM["ala_L"])[1])), "#1E90FF"),
    "Menton": ((0, int(y_pogo)), (W, int(y_pogo)), "#1E90FF"),
    "Midline": ((W // 2, 0), (W // 2, H), "#32CD32"),
    "Interincisale": ((int(P(LM["upper_incisor"])[0]), 0), (int(P(LM["upper_incisor"])[0]), H), "#FF1493"),
}

init_json = {"version": "4.4.0", "objects": []}
for name, ((x1, y1), (x2, y2), col) in init_lines.items():
    init_json["objects"].append({"type": "line", "name": name, "stroke": col,
                                 "strokeWidth": 2, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

st.write("### 1️⃣ Sposta le linee se necessario, poi premi **Ricalcola diagnosi**")
canvas = st_canvas(
    background_image=img_pil,
    initial_drawing=init_json,
    update_streamlit=True,
    height=H,
    width=W,
    drawing_mode="transform",
    key="canvas",
)

objs = {o["name"]: o for o in canvas.json_data["objects"]} if canvas.json_data else {}
line_y = lambda name: (objs[name]["y1"] + objs[name]["y2"]) / 2
line_x = lambda name: (objs[name]["x1"] + objs[name]["x2"]) / 2

if st.button("Ricalcola diagnosi"):
    try:
        y_T, y_G, y_S, y_M = map(line_y, ["Trichion", "Glabella", "Subnasale", "Menton"])
        x_mid = line_x("Midline")
        x_inc = line_x("Interincisale")
    except KeyError:
        st.error("Linee mancanti: ricarica pagina.")
        st.stop()

    H_tot = y_M - y_T
    thirds = np.array([(y_G - y_T), (y_S - y_G), (y_M - y_S)]) / H_tot * 100
    off_pct = abs(x_inc - x_mid) / W * 100
    sym_pct = np.sqrt(np.mean([abs(P(a)[0] - (W - P(b)[0]))**2 for a, b in PAIRS_SYMM])) / W * 100

    st.subheader("📊 Proporzioni (%)")
    st.json({"Sup": round(float(thirds[0]), 1), "Med": round(float(thirds[1]), 1), "Inf": round(float(thirds[2]), 1)})

    st.subheader("📐 Deviazioni (%)")
    st.json({"Midline": round(float(off_pct), 2), "Simmetria": round(float(sym_pct), 2)})

    # Diagnosi
    diag = []
    if np.all(np.abs(thirds - 33) < 3):
        diag.append("Proporzioni: normali (33 ± 3 %).")
    elif thirds[2] > ((thirds[0] + thirds[1]) / 2) * 1.1:
        diag.append("Long face: terzo inferiore aumentato.")
    elif thirds[2] < ((thirds[0] + thirds[1]) / 2) * 0.9:
        diag.append("Short face: terzo inferiore ridotto.")
    else:
        diag.append("Proporzioni: fuori range ideale.")

    diag.append("Midline ok" if off_pct < 0.4 else "Midline deviata (>0.4 %).")
    diag.append("Simmetria ok" if sym_pct < 3 else "Asimmetria (>3 %).")

    st.subheader("📝 Diagnosi")
    st.markdown("\n".join(["- " + d for d in diag]))

    st.subheader("🏷️ Range clinici")
    st.json(RANGE)

    st.caption("Valori in percentuale rispetto alle dimensioni facciali; confermare sempre con esame clinico.")
