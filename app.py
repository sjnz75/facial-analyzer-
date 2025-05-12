# Facial Aesthetic Analyzer – MVP
# --------------------------------------------------
# Autore: ChatGPT (OpenAI)
# Licenza: MIT
"""
Questo è un piccolo script Streamlit pensato per chi NON ha esperienza di coding. 
✅ Fa una prima analisi estetica da una foto frontale (midline, linea bipupillare, proporzioni in terzi)
🛠 Sezione profilo: placeholder che potrai completare in futuro o chiedermi di ampliare.

🔧 Dipendenze (da copiare in un file `requirements.txt`):
    streamlit
    mediapipe==0.10.8
    opencv-python
    pillow
    numpy

Come usare (zero‑code):
1. Crea un nuovo repo GitHub → aggiungi questo file col nome `app.py` + un file `requirements.txt` con le righe sopra.
2. Vai su https://streamlit.io/cloud → “New app” → collega il repo → Deploy. Fine!
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="Facial Aesthetic Analyzer", layout="centered")

st.title("📸 Facial Aesthetic Analyzer – MVP")

st.markdown("""
Carica una **foto frontale** e ottieni sovrimpressione di alcune linee/angoli di riferimento.
*(Versione di prova: per ora analizziamo solo la vista frontale; la vista laterale arriverà presto.)*
""")

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

front_file = st.file_uploader("🖼️ Carica foto frontale", type=["jpg", "jpeg", "png"])

if front_file:
    img = Image.open(front_file).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    res = mp_face.process(img_np)

    if not res.multi_face_landmarks:
        st.error("Volto non rilevato. Prova con un'immagine più chiara e frontale.")
        st.stop()

    lm = res.multi_face_landmarks[0].landmark
    P = lambda idx: np.array([lm[idx].x * w, lm[idx].y * h])

    # --- Calcoli semplici --------------------------------------------------
    mid_x = int((P(1)[0] + P(199)[0]) / 2)  # media di due landmark del naso
    left_eye = P(33).astype(int)            # angolo esterno occhio sinistro
    right_eye = P(263).astype(int)          # angolo esterno occhio destro
    glabella_y = int(P(9)[1])               # tra le sopracciglia
    subnasale_y = int(P(2)[1])              # base naso
    menton_y = int(P(152)[1])               # mento

    # --- Disegno sovrimpressioni ------------------------------------------
    annotated = img_np.copy()
    cv2.line(annotated, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)                 # midline (verde)
    cv2.line(annotated, tuple(left_eye), tuple(right_eye), (255, 0, 0), 2)       # linea bipupillare (blu)
    for y in [glabella_y, subnasale_y, menton_y]:                                # linee terzi (rosso)
        cv2.line(annotated, (0, y), (w, y), (0, 0, 255), 1)

    st.image(annotated, caption="Anteprima con linee di riferimento", use_column_width=True)

    # --- Output numerico ---------------------------------------------------
    data = {
        "distanza interpupillare (px)": float(np.linalg.norm(left_eye - right_eye)),
        "altezza terzo superiore (px)": abs(glabella_y - subnasale_y),
        "altezza terzo inferiore (px)": abs(subnasale_y - menton_y),
    }
    st.subheader("📐 Metriche grezze")
    st.json(data)

    st.caption(
        "⚠️ Questi valori sono in pixel. Per convertirli in millimetri serve un riferimento di scala (es. distanza interpupillare reale)."
    )

# ---------------------------------------------------------------------------
# TODO: modulo per foto di profilo
# ---------------------------------------------------------------------------
with st.expander("🚧 Roadmap / TODO"):
    st.markdown("""
* **Profilo laterale** con piani di Francoforte, Camper, E‑line.
* Calcolo automatico della linea del sorriso con immagini in movimento.
* Salvataggio report PDF.
* Interfaccia multilingua.
""")
