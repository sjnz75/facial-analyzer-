# Facial Aesthetic Analyzer – v0.6 (front-view, percentuali & diagnosi complete)
# ---------------------------------------------------------------------------
# Autore: ChatGPT (OpenAI) – 2025-05-12
# Licenza: MIT
"""
**Novità v0.6**
---------------
* Aggiunte linee **Trichion** e **Menton** (orizzontali estreme).
* Overlay con font ingrandito e contorno nero per **massima leggibilità**.
* Tutte le misure **in percentuale**: terzi facciali, simmetria, offset e allineamenti.
* Report diagnostico secondo i 5 criteri clinici:
  1. Linea mediana del volto
  2. Linea interincisale
  3. Simmetria facciale
  4. Linee verticali di riferimento
  5. Linee orizzontali di riferimento

📸 *Foto richiesta*: viso in NHP, fronte scoperta, occhi e commissure visibili.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Facial Analyzer v0.6", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v0.6")

st.markdown(
    "Carica una foto frontale in **NHP**; l'app disegnerà le linee di riferimento, calcolerà le proporzioni **in percentuale** e produrrà un **report diagnostico** chiaro.")

# Configurazione MediaPipe
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Landmark utili (MediaPipe 468 pts)
LM = {
    "trichion": 10,
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
}
PAIRS_SYMM = [(LM["eye_L"], LM["eye_R"]), (LM["brow_L"], LM["brow_R"]), (LM["mouth_L"], LM["mouth_R"]), (LM["ala_L"], LM["ala_R"])]

# Caricamento immagine
file = st.file_uploader("🖼️ Carica immagine JPG/PNG", type=["jpg","jpeg","png"])
if not file:
    st.stop()

# 1️⃣ Lettura e rotazione EXIF
img_pil = Image.open(file).convert("RGB")
img_pil = ImageOps.exif_transpose(img_pil)
img = np.array(img_pil)
h, w = img.shape[:2]

# 2️⃣ Landmark detection
res = mp_face.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato. Verifica illuminazione e posizione.")
    st.stop()
lm = res.multi_face_landmarks[0].landmark
P = lambda i: np.array([lm[i].x * w, lm[i].y * h])

# 3️⃣ Overlay leggibile
annotated = img.copy()
font = cv2.FONT_HERSHEY_DUPLEX
scale = 1.0
th_text = 2
th_border = 4

def put_label(img, text, org, color):
    cv2.putText(img, text, org, font, scale, (0,0,0), th_border, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, th_text, cv2.LINE_AA)

# Linee orizzontali (blu)
horiz = {
    "Trichion": int(P(LM["trichion"])[1]),
    "Bipupillare": int(P(LM["eye_L"])[1]),
    "Sopraccigliare": int(P(LM["brow_L"])[1]),
    "Commissurale": int(P(LM["mouth_L"])[1]),
    "Interalare": int(P(LM["ala_L"])[1]),
    "Menton": int(P(LM["pogonion"])[1])
}
for name,y in horiz.items():
    cv2.line(annotated, (0,y), (w,y), (255,0,0), 1)
    put_label(annotated, name, (10, y-10), (255,0,0))

# Linee verticali (arancio)
vert = {
    "Glabella": int(P(LM["glabella"])[0]),
    "Nasion": int(P(LM["nasion"])[0]),
    "Filtro labiale": int(P(LM["subnasale"])[0]),
    "Pogonion": int(P(LM["pogonion"])[0])
}
for name,x in vert.items():
    cv2.line(annotated, (x,0), (x,h), (0,165,255), 1)
    put_label(annotated, name, (x+5,40), (0,165,255))

# Midline (verde)
pts_mid = np.stack([P(LM["glabella"]), P(LM["subnasale"]), P(LM["pogonion"])]).astype(np.float32)
vx,vy,cx,cy = cv2.fitLine(pts_mid, cv2.DIST_L2,0,0.01,0.01)
t1 = (-cy)/vy if vy else 0
t2 = (h-cy)/vy if vy else 0
x1,y1 = int(cx+vx*t1),0
x2,y2 = int(cx+vx*t2),h
cv2.line(annotated,(x1,y1),(x2,y2),(0,255,0),2)
put_label(annotated, "Midline", (int(cx)+5,80), (0,255,0))

# Linea interincisale (magenta)
x_inc = int(P(LM["upper_incisor"])[0])
cv2.line(annotated,(x_inc,0),(x_inc,h),(255,0,255),1)
put_label(annotated, "Interincisale", (x_inc+5,110), (255,0,255))

# 4️⃣ Calcoli percentuali
# Terzi facciali
y_t = horiz["Trichion"]
y_m = horiz["Menton"]
tot = y_m - y_t
terzi = np.array([horiz["Sopraccigliare"]-y_t, horiz["Interalare"]-horiz["Sopraccigliare"], y_m - horiz["Interalare"]])
terzi_pct = terzi/tot*100
# Simmetria (% larghezza)
diffs = [abs(P(a)[0]-(w-P(b)[0])) for a,b in PAIRS_SYMM]
symm_pct = np.sqrt(np.mean(np.square(diffs)))/w*100
# Offset midline-incisale
mid_y = horiz["Interalare"]
x_mid = cx+vx*((mid_y-cy)/vy) if vy else cx
off_pct = abs(x_inc - x_mid)/w*100
# Orizzontali vs bipupillare (deviazione Y)
devs = [abs(y-horiz["Bipupillare"])/tot*100 for y in horiz.values()]
max_hdev = max(devs)
# Verticali vs midline (deviazione X)
voffs = [abs(x - cx)/w*100 for x in vert.values()]
max_voff = max(voffs)
# Angolo bipupillare
yL,yR = P(LM["eye_L"])[1],P(LM["eye_R"])[1]
ang = np.degrees(np.arctan2(yR-yL, P(LM["eye_R"])[0]-P(LM["eye_L"])[0]))

# 5️⃣ Visualizza risultato
st.image(annotated, caption="Overlay linee nominate", use_column_width=True)

st.subheader("📊 Proporzioni facciali (%)")
st.json({"Sup":round(terzi_pct[0],1),"Med":round(terzi_pct[1],1),"Inf":round(terzi_pct[2],1)})

st.subheader("📐 Allineamenti (%)")
st.json({
    "Offset Midline-Incisale":round(off_pct,2),
    "Simmetria globale":round(symm_pct,2),
    "Deviazione orizzontale max":round(max_hdev,2),
    "Deviazione verticale max":round(max_voff,2),
    "Angolo bipupillare (°)":round(ang,2)
})

st.subheader("📝 Report diagnostico secondo criteri clinici")
report=[]
# 1. Midline
if off_pct<0.4: report.append("• Midline coincidente (<0.4%).")
elif off_pct<0.8: report.append("• Midline lievemente deviato (0.4–0.8%).")
else: report.append("• Midline deviato (>0.8%).")
# 2. Interincisale
report.append(f"• Linea interincisale a {off_pct:.2f}% rispetto al centro facciale.")
# 3. Simmetria
if symm_pct<3: report.append("• Simmetria entro limiti clinici (<3%).")
else: report.append("• Simmetria percepibile (>3%).")
# 4. Verticali
if max_voff<3: report.append("• Linee verticali allineate (<3%).")
else: report.append(f"• Verticali disallineate (max {max_voff:.1f}%).")
# 5. Orizzontali
if max_hdev<2: report.append("• Linee orizzontali parallele (<2%).")
else: report.append(f"• Orizzontali disallineate (max {max_hdev:.1f}%).")

# Proporzioni terzi
up,md,low=terzi_pct
if low>(up+md)/2*1.1: report.append("• Long face.")
elif low<(up+md)/2*0.9: report.append("• Short face.")
else: report.append("• Proporzioni normali.")

st.markdown("\n".join(report))

st.caption("Guida clinica: percentuali riferite all'altezza/largezza facciale; validare con esame diretto.")
