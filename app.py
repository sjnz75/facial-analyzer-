# Facial Aesthetic Analyzer – v0.5 (front-view, full proportions & readable overlay)
# ---------------------------------------------------------------------------
# Autore: ChatGPT (OpenAI) – 2025-05-12
# Licenza: MIT
"""
**Novità v0.5**
---------------
* Aggiunte le linee **Trichion** (attaccatura capelli) e **Menton** (punto più inferiore del mento).
* Testo overlay ingrandito (font Duplex, contorno nero) ⇒ leggibile su qualsiasi sfondo.
* Tutte le misure ora in **percentuale** rispetto alla distanza Trichion-Menton.
* Report diagnostico aggiornato (Short / Normal / Long face in base ai terzi percentuali).

📸 *Foto richiesta*: viso frontale in NHP, fronte scoperta (necessario per individuare il **Trichion**).
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Facial Analyzer v0.5", layout="centered")

st.title("📸 Facial Aesthetic Analyzer – v0.5")

st.markdown(
    "Carica una foto **frontale** del paziente in **NHP** (natural head position) con fronte visibile.\n"
    "Lʼapp disegnerà tutte le linee nominate, calcolerà le proporzioni tra i tre terzi facciali in **percentuale** e genererà un report clinico.")

# ---------------------------------------------------------------------------
# Config MediaPipe
# ---------------------------------------------------------------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

# Landmark ID utili (MediaPipe 468-pts)
LM = {
    "trichion": 10,            # approx. hairline center
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

file = st.file_uploader("🖼️ Carica foto (jpg/png)", type=["jpg", "jpeg", "png"])

if file:
    img_pil = Image.open(file).convert("RGB")
    img_pil = ImageOps.exif_transpose(img_pil)
    img = np.array(img_pil)
    h, w = img.shape[:2]

    res = mp_face.process(img)
    if not res.multi_face_landmarks:
        st.error("Volto non rilevato. Controlla illuminazione e posizione.")
        st.stop()

    lm = res.multi_face_landmarks[0].landmark
    P = lambda idx: np.array([lm[idx].x * w, lm[idx].y * h])

    # ---------- Overlay helpers ----------
    font = cv2.FONT_HERSHEY_DUPLEX
    def put_label(img, text, org, color):
        cv2.putText(img, text, org, font, 0.7, (0,0,0), 4, cv2.LINE_AA)  # contorno nero
        cv2.putText(img, text, org, font, 0.7, color, 2, cv2.LINE_AA)

    annotated = img.copy()

    # ----- Orizzontali (blu) -----
    horiz = {
        "Linea Trichion": int(P(LM["trichion"])[1]),
        "Linea bipupillare": int(P(LM["eye_L"])[1]),
        "Linea sopraccigliare": int(P(LM["brow_L"])[1]),
        "Linea commissurale": int(P(LM["mouth_L"])[1]),
        "Linea interalare": int(P(LM["ala_L"])[1]),
        "Linea Menton": int(P(LM["pogonion"])[1]),
    }
    for name, y in horiz.items():
        cv2.line(annotated, (0, y), (w, y), (255, 0, 0), 1)
        put_label(annotated, name, (10, y - 8), (255, 0, 0))

    # ----- Verticali (arancio) -----
    vert = {
        "Glabella": int(P(LM["glabella"])[0]),
        "Nasion": int(P(LM["nasion"])[0]),
        "Filtro labiale": int(P(LM["subnasale"])[0]),
        "Pogonion": int(P(LM["pogonion"])[0]),
    }
    for name, x in vert.items():
        cv2.line(annotated, (x, 0), (x, h), (0, 165, 255), 1)
        put_label(annotated, name, (x + 5, 20), (0, 165, 255))

    # ----- Midline (verde) -----
    pts_mid = np.stack([P(LM["glabella"]), P(LM["subnasale"]), P(LM["pogonion"])]).astype(np.float32)
    vx, vy, cx, cy = cv2.fitLine(pts_mid, cv2.DIST_L2, 0, 0.01, 0.01)
    t1 = (-cy) / vy if vy != 0 else 0
    t2 = (h - cy) / vy if vy != 0 else 0
    x1, y1 = int(cx + vx * t1), 0
    x2, y2 = int(cx + vx * t2), h
    cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    put_label(annotated, "Midline", (int(cx)+5, 40), (0, 255, 0))

    # ----- Linea interincisale (magenta) -----
    inc_x = int(P(LM["upper_incisor"])[0])
    cv2.line(annotated, (inc_x, 0), (inc_x, h), (255, 0, 255), 1)
    put_label(annotated, "Linea interincisale", (inc_x + 5, 60), (255, 0, 255))

    # ---------- Proporzioni (percentuale) ----------
    y_trich = horiz["Linea Trichion"]
    y_brow = horiz["Linea sopraccigliare"]
    y_alar = horiz["Linea interalare"]
    y_ment = horiz["Linea Menton"]

    total = y_ment - y_trich
    thirds_px = np.array([y_brow - y_trich, y_alar - y_brow, y_ment - y_alar])
    thirds_pct = thirds_px / total * 100

    # ---------- Simmetria (% larghezza) ----------
    diffs = []
    for a, b in PAIRS_SYMM:
        xa, ya = P(a)
        xb, yb = P(b)
        diffs.append(abs(xa - (w - xb)))
    symmetry_pct = np.sqrt(np.mean(np.square(diffs))) / w * 100

    # ---------- Diagnosi ----------
    up, mid, low = thirds_pct
    if low > (up + mid)/2 * 1.1:
        face_type = "Long face (terzo inferiore aumentato)"
    elif low < (up + mid)/2 * 0.9:
        face_type = "Short face (terzo inferiore ridotto)"
    else:
        face_type = "Proporzioni nei limiti (normal face)"

    off_px = abs(inc_x - (cx + vx*((y_alar - cy)/vy) if vy!=0 else cx))
    off_pct = off_px / w * 100

    angle_pupil = np.degrees(np.arctan2(P(LM["eye_R"])[1] - P(LM["eye_L"])[1], P(LM["eye_R"])[0] - P(LM["eye_L"])[0]))

    # ---------- Output ----------
    st.image(annotated, caption="Overlay con linee nominate", use_column_width=True)

    st.subheader("📊 Proporzioni facciali (%)")
    st.json({
        "Terzo superiore": round(up,1),
        "Terzo medio": round(mid,1),
        "Terzo inferiore": round(low,1),
    })

    st.subheader("📐 Altri indici (%)")
    st.json({
        "Offset midline-incisale": round(off_pct,2),
        "Indice RMS simmetria": round(symmetry_pct,2),
        "Angolo bipupillare (°)": round(angle_pupil,2),
    })

    st.subheader("📝 Report diagnostico")
    report = []
    report.append(f"• {face_type}.")

    if off_pct < 0.4:  # ~2mm su volto 500px
        report.append("• Midline dentale coincidente (<0.4 % larghezza volto).")
    elif off_pct < 0.8:
        report.append("• Midline dentale lievemente deviata (0.4–0.8 %).")
    else:
        report.append("• Midline dentale deviata (>0.8 %): correzione consigliata.")

    if abs(angle_pupil) < 2:
        report.append("• Linea bipupillare orizzontale (≤2°).")
    else:
        report.append(f"• Inclinazione bipupillare di {angle_pupil:.1f}°: valutare compensi posturali.")

    if symmetry_pct < 3:
        report.append("• Simmetria facciale entro limiti clinici (<3 %).")
    else:
        report.append("• Asimmetria facciale percepibile (>3 %).")

    st.markdown("\n".join(report))

    st.caption("Le misure sono percentuali rispetto allʼaltezza e larghezza del volto; utilizzare come guida clinica, non come diagnosi definitiva.")
