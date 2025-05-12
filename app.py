# Facial Aesthetic Analyzer – v0.4 (front-view, scientific & diagnostic)
# ---------------------------------------------------------------------------
# Autore: ChatGPT (OpenAI) 2025-05-12
# Licenza: MIT
"""
**Funzioni implementate**
------------------------
* Analisi estetica frontale completa (5 linee di riferimento ufficiali).
* Overlay con **nomi** delle linee direttamente sull’immagine.
* Conversione **pixel → millimetri** usando la distanza interpupillare reale inserita dall’utente.
* **Report diagnostico** automatico secondo criteri clinici (midline, linee orizzontali, proporzioni facciali).

> ⚠️ Fotografia richiesta: viso in **NHP** (Natural Head Position), luce frontale, incisivi centrali visibili, capelli scostati dalla fronte.

Dipendenze (requirements.txt)
-----------------------------
```
streamlit==1.34.0
mediapipe==0.10.21
opencv-python-headless==4.9.0.80
pillow
numpy
```
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Facial Analyzer v0.4", layout="centered")

st.title("📸 Facial Aesthetic Analyzer – v0.4")

st.markdown(
    "Carica una foto **frontale** del paziente in NHP, poi inserisci la **distanza interpupillare reale** (mm).\n"
    "Lʼapp disegnerà le linee di riferimento col loro nome, calcolerà le distanze in millimetri e genererà un breve **report diagnostico**.")

# ---------------------------------------------------------------------------
# Config MediaPipe
# ---------------------------------------------------------------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

# ---------------------------------------------------------------------------
# Landmark ID utili (MediaPipe Face Mesh 468-pts)
# ---------------------------------------------------------------------------
LM = {
    "trichion": 10,            # top-of-forehead approx. (visibile solo se capelli scoperti)
    "glabella": 9,
    "nasion": 168,
    "subnasale": 2,
    "pogonion": 152,
    "eye_L": 33,
    "eye_R": 263,
    "brow_L": 70,
    "brow_R": 300,
    "mouth_L": 61,
    "mouth_R": 291,
    "ala_L": 98,
    "ala_R": 327,
    "upper_incisor": 13,
    "lower_incisor": 14,
}

PAIRS_SYMM = [
    (LM["eye_L"], LM["eye_R"]),
    (LM["brow_L"], LM["brow_R"]),
    (LM["mouth_L"], LM["mouth_R"]),
    (LM["ala_L"], LM["ala_R"]),
]

# ---------------------------------------------------------------------------
# Upload + parametri utente
# ---------------------------------------------------------------------------
file = st.file_uploader("🖼️ Carica foto (jpg/png)", type=["jpg", "jpeg", "png"])
real_ipd_mm = st.number_input("Inserisci distanza interpupillare reale (mm)", min_value=40.0, max_value=80.0, value=63.0, step=0.1)

if file:
    # 1️⃣ Leggi immagine rispettando EXIF
    img_pil = Image.open(file).convert("RGB")
    img_pil = ImageOps.exif_transpose(img_pil)
    img = np.array(img_pil)
    h, w = img.shape[:2]

    # 2️⃣ Landmark detection
    res = mp_face.process(img)
    if not res.multi_face_landmarks:
        st.error("Volto non rilevato. Controlla illuminazione e posizione.")
        st.stop()

    lm = res.multi_face_landmarks[0].landmark
    P = lambda idx: np.array([lm[idx].x * w, lm[idx].y * h])

    # 3️⃣ Pixel → millimetri scale factor
    ipd_px = np.linalg.norm(P(LM["eye_L"]) - P(LM["eye_R"]))
    px_to_mm = real_ipd_mm / ipd_px if ipd_px > 0 else 0

    # 4️⃣ Costruisci overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    annotated = img.copy()

    def put_text(img, text, origin, color=(255, 255, 255)):
        cv2.putText(img, text, origin, font, 0.6, color, 2, cv2.LINE_AA)

    # ---- Linee orizzontali (blu)
    horiz = {
        "Linea bipupillare": int(P(LM["eye_L"])[1]),
        "Linea sopraccigliare": int(P(LM["brow_L"])[1]),
        "Linea commissurale": int(P(LM["mouth_L"])[1]),
        "Linea interalare": int(P(LM["ala_L"])[1]),
    }
    for name, y in horiz.items():
        cv2.line(annotated, (0, y), (w, y), (255, 0, 0), 1)
        put_text(annotated, name, (10, y - 5))

    # ---- Linee verticali (arancio)
    vert = {
        "Glabella": int(P(LM["glabella"])[0]),
        "Nasion": int(P(LM["nasion"])[0]),
        "Filtro labiale": int(P(LM["subnasale"])[0]),
        "Pogonion": int(P(LM["pogonion"])[0]),
    }
    for name, x in vert.items():
        cv2.line(annotated, (x, 0), (x, h), (0, 165, 255), 1)
        put_text(annotated, name, (x + 5, 20))

    # ---- Midline best-fit (verde)
    pts_mid = np.stack([P(LM["glabella"]), P(LM["subnasale"]), P(LM["pogonion"])]).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts_mid, cv2.DIST_L2, 0, 0.01, 0.01)
    def draw_inf_line(vx, vy, cx, cy, color=(0, 255, 0), thickness=2):
        t1 = (-cy) / vy if vy != 0 else 0
        t2 = (h - cy) / vy if vy != 0 else 0
        x1, y1 = int(cx + vx * t1), 0
        x2, y2 = int(cx + vx * t2), h
        cv2.line(annotated, (x1, y1), (x2, y2), color, thickness)
    draw_inf_line(vx, vy, cx, cy)
    put_text(annotated, "Midline", (int(cx) + 5, 40), (0, 255, 0))

    # ---- Linea interincisale (magenta)
    inc_x = int(P(LM["upper_incisor"])[0])
    cv2.line(annotated, (inc_x, 0), (inc_x, h), (255, 0, 255), 1)
    put_text(annotated, "Linea interincisale", (inc_x + 5, 60), (255, 0, 255))

    # 5️⃣ Distanze verticali (mm)
    y_brow = horiz["Linea sopraccigliare"]
    y_alar = horiz["Linea interalare"]
    y_menton = int(P(LM["pogonion"])[1])  # menton ~ pogonion y

    third_upper_px = y_brow - vert["Glabella"]
    third_middle_px = y_alar - y_brow
    third_lower_px = y_menton - y_alar

    thirds_mm = np.array([third_upper_px, third_middle_px, third_lower_px]) * px_to_mm

    # 6️⃣ Simmetria (RMS)
    diffs = []
    for a, b in PAIRS_SYMM:
        xa, ya = P(a)
        xb, yb = P(b)
        diffs.append(abs(xa - (w - xb)))  # semplice specchio rispetto metà immagine
    symmetry_mm = np.sqrt(np.mean(np.square(diffs))) * px_to_mm

    # 7️⃣ Diagnosi proporzioni
    def classify_face(thirds):
        up, mid, low = thirds
        mean_ref = (up + mid) / 2
        if low > mean_ref * 1.1:
            return "Long face (terzo inferiore aumentato)"
        elif low < mean_ref * 0.9:
            return "Short face (terzo inferiore ridotto)"
        else:
            return "Proporzioni nei limiti (normal face)"
    face_diag = classify_face(thirds_mm)

    # 8️⃣ Midline vs interincisale offset
    mid_x = cx + vx * ((y_alar - cy) / vy) if vy != 0 else cx  # x midline a livello interalare
    off_mm = abs((inc_x - mid_x) * px_to_mm)

    # 9️⃣ Inclinazione bipupillare
    eye_L = P(LM["eye_L"])
    eye_R = P(LM["eye_R"])
    angle_pupil = np.degrees(np.arctan2(eye_R[1] - eye_L[1], eye_R[0] - eye_L[0]))

    # 10️⃣ Genera report
    st.image(annotated, caption="Overlay con linee nominative", use_column_width=True)

    st.subheader("📐 Dati metrici (millimetri)")
    st.json({
        "Terzo superiore (Tr-Brow)": round(thirds_mm[0], 1),
        "Terzo medio (Brow-Alar)": round(thirds_mm[1], 1),
        "Terzo inferiore (Alar-Menton)": round(thirds_mm[2], 1),
        "Offset midline-incisale": round(off_mm, 1),
        "Indice RMS simmetria": round(symmetry_mm, 2),
        "Angolo linea bipupillare (°)": round(angle_pupil, 2),
    })

    st.subheader("📝 Report diagnostico")
    report = []
    # Proporzioni facciali
    report.append(f"• {face_diag}.")
    # Midline
    if off_mm <= 2:
        report.append("• Midline dentale coincidente (≤2 mm).")
    elif off_mm <= 4:
        report.append("• Midline dentale lievemente deviata (2-4 mm). Potrebbe non essere percepita.")
    else:
        report.append("• Midline dentale deviata (>4 mm): correzione consigliata.")
    # Bipupillare
    if abs(angle_pupil) < 2:
        report.append("• Linea bipupillare orizzontale (≤2°).")
    else:
        report.append("• Inclinazione bipupillare di {:.1f}°: valutare compensi posturali.".format(angle_pupil))
    # Simmetria globale
    if symmetry_mm * 1000 / (w * px_to_mm) < 3:  # %<3
        report.append("• Simmetria facciale entro limiti clinici (<3%).")
    else:
        report.append("• Asimmetria facciale percepibile (>3%).")

    st.markdown("\n".join(report))

    st.caption("I risultati sono generati automaticamente da immagini 2D; un esame clinico approfondito resta indispensabile.")
