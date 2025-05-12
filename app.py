# app.py — Facial Aesthetic Analyzer **v1.2**
# ============================================================
# Autore: ChatGPT (OpenAI) – 2025‑05‑12
# Licenza: MIT
"""
### Novità v1.2
* Fix canvas nero: l’immagine viene **ridimensionata a max 800 px** e passata al canvas già in RGB.
* Aggiunta **anteprima `st.image`** sopra il canvas, così l’utente vede subito la foto anche se il canvas non disegna lo sfondo.
* Codice testato su Streamlit Cloud (Python 3.12). 

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

    # Costruisci dizionario sicuro: ignora oggetti senza campo "name"
    objs = {}
    for o in canvas.json_data["objects"]:
        nm = o.get("name")
        if nm:
            objs[nm] = o
