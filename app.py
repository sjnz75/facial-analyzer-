# Facial Aesthetic Analyzer – MVP 0.2
# --------------------------------------------------
# Autore: ChatGPT (OpenAI)
# Licenza: MIT
"""
CHANGELOG v0.2
---------------
* 🖼 **Exif-auto-rotation**: la foto ora viene ruotata automaticamente in verticale.
* 📐 Fix calcoli terzi facciali (superiore, medio, inferiore).
* 🏷 Output in pixel + proporzioni (%), più facile da interpretare.

ISTRUZIONI (immutate)
---------------------
1. Repo GitHub → file `app.py` + `requirements.txt`.
2. Streamlit Cloud → Deploy.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Facial Aesthetic Analyzer", layout="centered")

st.title("📸 Facial Aesthetic Analyzer – MVP 0.2")

st.markdown(
    """
Carica una **foto frontale** ben illuminata. L'app disegnerà:
* linea mediana (verde)
* linea bipupillare (blu)
* terzi facciali superiore / medio / inferiore (rosso)

_Esif autoprotate attivo: puoi caricare la foto così com'è; se hai ancora problemi di orientamento, fammelo sapere._
"""
)

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

front_file = st.file_uploader("🖼️ Carica foto frontale", type=["jpg", "jpeg", "png"])

if front_file:
    # 1️⃣ Leggi immagine e ruota in base all'EXIF (se necessario)
    img_in = Image.open(front_file).convert("RGB")
    img_rot = ImageOps.exif_transpose(img_in)  # rispetta orientamento camera
    img_np = np.array(img_rot)
    h, w = img_np.shape[:2]

    res = mp_face.process(img_np)

    if not res.multi_face_landmarks:
        st.error("Volto non rilevato. Prova con un'immagine più chiara e frontale.")
        st.stop()

    lm = res.multi_face_landmarks[0].landmark
    P = lambda idx: np.array([lm[idx].x * w, lm[idx].y * h])

    # Landmark usati → vedi docs MediaPipe Face Mesh
    L_GLAB = 9       # tra le sopracciglia
    L_SUBN = 2       # subnasale (base naso)
    L_MENT = 152     # menton (pogonion)
    L_EYE_L = 33     # angolo occhio sinistro (utente)
    L_EYE_R = 263    # angolo occhio destro (utente)
    L_NOSE = 1       # centro columella, per midline
    L_NOSE_R = 199   # punto simmetrico per midline (narice dx)

    # 2️⃣ Calcoli geometrici ---------------------------------------------
    mid_x = int(np.mean([P(L_NOSE)[0], P(L_NOSE_R)[0]]))  # x verticale midline

    # Linea bipupillare → passa per i due angoli esterni
    left_eye = P(L_EYE_L).astype(int)
    right_eye = P(L_EYE_R).astype(int)

    # Terzi facciali (verticali)
    y_glab = P(L_GLAB)[1]
    y_subn = P(L_SUBN)[1]
    y_ment = P(L_MENT)[1]

    third_sup = y_subn - y_glab       # px
    third_inf = y_ment - y_subn       # px
    third_total = y_ment - y_glab
    third_mid = third_total - third_sup - third_inf  # placeholder se servisse capello→glab

    # 3️⃣ Disegno overlay --------------------------------------------------
    annotated = img_np.copy()
    # midline
    cv2.line(annotated, (mid_x, 0), (mid_x, h), (0, 255, 0), 2)
    # bipupillare
    cv2.line(annotated, tuple(left_eye), tuple(right_eye), (255, 0, 0), 2)
    # terzi
    for y in [int(y_glab), int(y_subn), int(y_ment)]:
        cv2.line(annotated, (0, int(y)), (w, int(y)), (0, 0, 255), 1)

    st.image(annotated, caption="Anteprima con linee di riferimento", use_column_width=True)

    # 4️⃣ Report numerico --------------------------------------------------
    report_px = {
        "interpupillary_dist_px": float(np.linalg.norm(left_eye - right_eye)),
        "third_upper_px": float(third_sup),
        "third_lower_px": float(third_inf),
    }

    # Percentuale rispetto al totale glabella→menton
    report_pct = {
        "third_upper_%": round(100 * third_sup / third_total, 1),
        "third_lower_%": round(100 * third_inf / third_total, 1),
    }

    st.subheader("📐 Metriche (pixel)")
    st.json(report_px)
    st.subheader("📊 Proporzioni (%)")
    st.json(report_pct)

    st.caption("⚠️ Valori in pixel e percentuale; per mm necessita un riferimento di scala (es. righello o distanza interpupillare reale).")

# ---------------------------------------------------------------------------
# TODO: Analisi laterale, linea del sorriso, export PDF
# ---------------------------------------------------------------------------
with st.expander("🚧 Roadmap / TODO"):
    st.markdown(
        """
* **Profilo laterale** con piani di Francoforte, Camper, E-line.
* Calcolo automatico della linea del sorriso con immagini in movimento.
* Salvataggio report PDF.
* Interfaccia multilingua.
"""
    )
