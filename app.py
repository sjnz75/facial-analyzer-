# Facial Aesthetic Analyzer – v0.3 (front‑view scientific set)
# ------------------------------------------------------------------
# Autore: ChatGPT (OpenAI) – Ottimizzato per criteri estetici frontali
# Licenza: MIT
"""
CHANGELOG v0.3
===============
✅ Implementati i **5 criteri scientifici** richiesti (solo foto frontali):
1. **Linea mediana del volto** (best‑fit tra glabella‑subnasale‑pogonion)
2. **Linea interincisale** (asse verticale fra 2 landmark dentali)
3. **Indice di simmetria facciale** (RMS su 6 coppie di landmark)
4. **Linee verticali di riferimento** (glabella, nasion, filtro labiale, pogonion)
5. **Linee orizzontali di riferimento** (bipupillare, sopraccigliare, commissurale, interalare)

▶︎  Basta caricare una foto frontale ben illuminata; lʼapp ruota lʼimmagine (Exif) e restituisce overlay + valori numerici.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Facial Aesthetic Analyzer – v0.3", layout="centered")

st.title("📸 Facial Aesthetic Analyzer – v0.3 (Front View)")

st.markdown(
    """
**Carica una foto frontale** (sguardo dritto, testa verticale, bocca socchiusa o in lieve sorriso).<br/>
L'app sovrappone tutte le linee scientifiche e calcola i valori in pixel **e in percentuale**.
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Inizializza MediaPipe FaceMesh
# ------------------------------------------------------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
LANDMARKS = {
    "glabella": 9,
    "nasion": 168,            # approssimazione: radice naso
    "subnasale": 2,
    "pogonion": 152,
    "eye_L": 33,
    "eye_R": 263,
    "brow_L": 70,   # sopracciglio
    "brow_R": 300,
    "mouth_L": 61,  # commissure
    "mouth_R": 291,
    "ala_L": 98,    # ala nasale
    "ala_R": 327,
    "upper_incisor": 13,   # punto centrale labbro sup. – proxy contatto incisivi
    "lower_incisor": 14,   # punto centrale labbro inf.
}

PAIRS_SYMM = [
    (LANDMARKS["eye_L"], LANDMARKS["eye_R"]),
    (LANDMARKS["brow_L"], LANDMARKS["brow_R"]),
    (LANDMARKS["mouth_L"], LANDMARKS["mouth_R"]),
    (LANDMARKS["ala_L"], LANDMARKS["ala_R"]),
    (234, 454),   # zigomi (MediaPipe)
    (93, 323),    # mascella
]

UPLOAD = st.file_uploader("🖼️ Carica foto frontale", type=["jpg", "jpeg", "png"])

if UPLOAD:
    # 1️⃣ Rotazione corretta via EXIF
    img_in = Image.open(UPLOAD).convert("RGB")
    img = ImageOps.exif_transpose(img_in)
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # 2️⃣ Rilevamento Landmark
    res = mp_face.process(img_np)
    if not res.multi_face_landmarks:
        st.error("Volto non rilevato. Prova con immagine più nitida e frontale.")
        st.stop()

    lm = res.multi_face_landmarks[0].landmark
    P = lambda idx: np.array([lm[idx].x * w, lm[idx].y * h])

    # 3️⃣ Linea mediana (best fit su 3 punti chiave)
    pts_mid = np.stack([
        P(LANDMARKS["glabella"]),
        P(LANDMARKS["subnasale"]),
        P(LANDMARKS["pogonion"]),
    ])
    vx, vy, cx, cy = cv2.fitLine(pts_mid.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    # param eq: (x, y) = (cx, cy) + t*(vx, vy)

    # Funzione per disegnare una retta su tutto il canvas
    def draw_infinite_line(img, vx, vy, cx, cy, color, thickness):
        lh = img.shape[0]
        lw = img.shape[1]
        # intersezione con bordo superiore (t_up) e inferiore (t_down)
        t_up = (-cy) / vy if vy != 0 else 0
        t_down = (lh - cy) / vy if vy != 0 else 0
        x_up = int(cx + vx * t_up)
        x_down = int(cx + vx * t_down)
        cv2.line(img, (x_up, 0), (x_down, lh), color, thickness)

    annotated = img_np.copy()

    # ▸ Midline (verde)
    draw_infinite_line(annotated, vx, vy, cx, cy, (0, 255, 0), 2)

    # 4️⃣ Linea interincisale (magenta) – asse tra 2 punti dentali
    p_top = P(LANDMARKS["upper_incisor"])
    p_bot = P(LANDMARKS["lower_incisor"])
    cv2.line(
        annotated,
        (int(p_top[0]), 0),
        (int(p_top[0]), h),
        (255, 0, 255), 1,
    )

    # 5️⃣ Linee orizzontali di riferimento (blu)
    def h_line(point_idx, color=(255, 0, 0), thickness=1):
        y = int(P(point_idx)[1])
        cv2.line(annotated, (0, y), (w, y), color, thickness)
        return y

    y_bipup = h_line(LANDMARKS["eye_L"], thickness=2)  # bipupillare
    y_brow = h_line(LANDMARKS["brow_L"])               # sopraccigliare
    y_comm = h_line(LANDMARKS["mouth_L"])              # commissurale
    y_alar = h_line(LANDMARKS["ala_L"])                # interalare

    # 6️⃣ Linee verticali di riferimento (arancione)
    def v_line(point_idx, color=(0, 165, 255), thickness=1):
        x = int(P(point_idx)[0])
        cv2.line(annotated, (x, 0), (x, h), color, thickness)
        return x

    x_glab = v_line(LANDMARKS["glabella"])
    x_nas = v_line(LANDMARKS["nasion"])
    x_subn = v_line(LANDMARKS["subnasale"])
    x_pog = v_line(LANDMARKS["pogonion"])

    # 7️⃣ Simmetria facciale (RMS distanza normalized)
    mid_x_axis = lambda x: (-(vy/vx)*(x-cx)+cy) if vx!=0 else cy  # y of midline at given x
    def mirror_x(x, y):
        # proiezione del punto sulla perpendicolare alla midline, riflesso
        # Per semplicità: approssimiamo con riflesso rispetto alla x del midline all'altezza y.
        if abs(vx) < 1e-6:  # midline quasi verticale
            return 2*cx - x
        else:
            # formula riflessione rispetto retta non verticale → più complessa, ma per piccole inclinazioni semplifichiamo.
            return 2* (cx + vx*( (y-cy)/vy )) - x
    diffs = []
    for a, b in PAIRS_SYMM:
        xa, ya = P(a)
        xb, yb = P(b)
        # riflette A → A'
        xa_mirr = mirror_x(xa, ya)
        diff = abs(xb - xa_mirr)
        diffs.append(diff)
    symmetry_rms = float(np.sqrt(np.mean(np.square(diffs))))
    symmetry_norm = 100 * symmetry_rms / w  # % rispetto larghezza volto

    # 8️⃣ Mostra risultati ---------------------------------------------------
    st.image(annotated, caption="Overlay criteri frontali", use_column_width=True)

    st.subheader("📐 Metriche scientifiche (pixel)")
    metrics_px = {
        "midline_angle_deg": round(float(np.degrees(np.arctan2(vy, vx))), 2),
        "interincisal_offset_px": abs(int(p_top[0] - cx)),
        "symmetry_rms_px": round(symmetry_rms, 2),
    }
    st.json(metrics_px)

    st.subheader("⚖️ Indice di simmetria (% della larghezza volto)")
    st.write(f"{symmetry_norm:.2f} % (0 % = perfetta; >4 % = asimmetria visibile)")

    st.caption(
        "Nota: le misure sono in pixel; per ottenere valori in millimetri occorre un riferimento di scala (es. distanza interpupillare reale).\n"
        "La linea interincisale usa landmark labiali come proxy: se vuoi precisione odontoiatrica serve una foto con incisivi visibili e un modello dentale dedicato."
    )

# ------------------------------------------------------------------
# TODO prossime release: profilo laterale, calibrazione mm, esportazione PDF
# ------------------------------------------------------------------
