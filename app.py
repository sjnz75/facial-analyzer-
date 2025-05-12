# Facial Aesthetic Analyzer – v0.7 (front-view, precise Trichion)
# ---------------------------------------------------------------------------
# Autore: ChatGPT (OpenAI) – 2025-05-12
# Licenza: MIT
"""
**Novità v0.7**
---------------
* Calcolo dinamico della **linea di staccatura capelli (Trichion)** sulla base della proporzione glabella–subnasale.
* Corretto posizionamento del Menton (punto inferiore terzo).
* Overlay ingrandito, con contorno nero, leggibile.
* Tutte le misure in **percentuale** rispetto all’altezza facciale (Trichion→Menton) o alla larghezza (
  simmetria, offset) a seconda dei criteri.
* Report diagnostico sui 5 parametri clinici richiesti.

📸 *Foto richiesta*: viso frontale in NHP, fronte scoperta, occhi e commissure visibili.
"""

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Facial Analyzer v0.7", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v0.7")

st.markdown(
    "Carica una foto frontale in **NHP**; l'app disegnerà linee di riferimento, calcolerà le proporzioni **in percentuale** e genererà un report diagnostico." 
)

# Inizializza FaceMesh
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Landmark utili
LM = {
    "glabella": 9,
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

# 1️⃣ Leggi e ruota
img_pil = Image.open(file).convert("RGB")
img_pil = ImageOps.exif_transpose(img_pil)
img = np.array(img_pil)
h, w = img.shape[:2]

# 2️⃣ Landmark detection
res = mp_face.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato. Verifica posizionamento e illuminazione.")
    st.stop()
lm = res.multi_face_landmarks[0].landmark
P = lambda i: np.array([lm[i].x * w, lm[i].y * h])

# Calcola coordinate chiave
y_glab = P(LM["glabella"])[1]
y_subn = P(LM["subnasale"])[1]
y_pogo = P(LM["pogonion"])[1]
# Trichion calcolato: riflessione di subnasale->glabella
y_trich = max(0, int(2*y_glab - y_subn))

# Prepara overlay
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
    "Trichion": y_trich,
    "Glabella": int(y_glab),
    "Subnasale": int(y_subn),
    "Commissurale": int(P(LM["mouth_L"])[1]),
    "Interalare": int(P(LM["ala_L"])[1]),
    "Menton": int(y_pogo),
}
for name, y in horiz.items():
    cv2.line(annotated, (0,y), (w,y), (255,0,0), 1)
    put_label(annotated, name, (10, y-10), (255,0,0))

# Linee verticali (arancio)
vert = {
    "Glabella": int(P(LM["glabella"])[0]),
    "Nasion": int(P(LM["glabella"])[0]),  # sostituisce nasion basandosi su glabella x
    "Filtro labiale": int(P(LM["subnasale"])[0]),
    "Pogonion": int(P(LM["pogonion"])[0])
}
for name, x in vert.items():
    cv2.line(annotated, (x,0), (x,h), (0,165,255), 1)
    put_label(annotated, name, (x+5,40), (0,165,255))

# Midline (verde)
pts_mid = np.stack([ [w/2,y_glab], [w/2,y_subn], [w/2,y_pogo] ]).astype(np.float32)
# uso asse verticale centrale: midline x=w/2
cv2.line(annotated, (w//2,0), (w//2,h), (0,255,0), 2)
put_label(annotated, "Midline", (w//2+5,80), (0,255,0))

# Linea interincisale (magenta)
x_inc = int(P(LM["upper_incisor"])[0])
cv2.line(annotated,(x_inc,0),(x_inc,h),(255,0,255),1)
put_label(annotated, "Interincisale", (x_inc+5,110), (255,0,255))

# 4️⃣ Calcoli percentuali
# Terzi facciali: suddivido Trichion->Glabella, Glabella->Subnasale, Subnasale->Menton in 3 segmenti
vals = [ horiz["Glabella"]-horiz["Trichion"], horiz["Subnasale"]-horiz["Glabella"], horiz["Menton"]-horiz["Subnasale"] ]
thirds_pct = [v/(horiz["Menton"]-horiz["Trichion"] )*100 for v in vals]
# Simmetria (% larghezza): RMS specchio rispetto a metà immagine
diffs = [abs(P(a)[0]-(w-P(b)[0])) for a,b in PAIRS_SYMM]
symm_pct = np.sqrt(np.mean(np.square(diffs)))/w*100
# Offset midline-incisale
off_pct = abs(x_inc - w/2)/w*100
# Deviazioni linee orizzontali
h_devs = [abs(y-horiz["Bipupillare"])/ (horiz["Menton"]-horiz["Trichion"]) *100 for y in horiz.values()]
max_hdev = max(h_devs)
# Deviazioni verticali
v_devs = [abs(x-w/2)/w*100 for x in vert.values()]
max_vdev = max(v_devs)
# Angolo bipupillare
yL, xL = P(LM["eye_L"])[1], P(LM["eye_L"])[0]
arg = P(LM["eye_R"])
yR, xR = arg[1], arg[0]
ang = np.degrees(np.arctan2(yR-yL, xR-xL))

# 5️⃣ Output
st.image(annotated, caption="Overlay linee nominate", use_column_width=True)

st.subheader("📊 Proporzioni facciali (%)")
st.json({"Sup":round(thirds_pct[0],1),"Med":round(thirds_pct[1],1),"Inf":round(thirds_pct[2],1)})

st.subheader("📐 Allineamenti (%)")
st.json({
    "Off Midline-Incisale":round(off_pct,2),
    "Simmetria":round(symm_pct,2),
    "Dev orizzont.:":round(max_hdev,2),
    "Dev vert.:":round(max_vdev,2),
    "Ang bipupillare (°)":round(ang,2)
})

st.subheader("📝 Report diagnostico secondo criteri clinici")
rep=[]
# 1. Midline
rep.append("Midline coincid%s (<0.5%%)." % ("ente" if off_pct<0.5 else " deviata")).
# 2. Interincisale
tmp = f"Interincisale a {off_pct:.2f}%% dal centro facciale."; rep.append(tmp)
# 3. Simmetria
rep.append("Simmetria entro limiti (<3%%)." if symm_pct<3 else "Asimmetria (>3%%).")
# 4. Verticali
rep.append("Verticali allineate (<3%%)." if max_vdev<3 else f"Vert disallineate ({max_vdev:.1f}%%).")
# 5. Orizzontali
rep.append("Orizzontali parallele (<2%%)." if max_hdev<2 else f"Orizz disallineate ({max_hdev:.1f}%%).")
# Proporzioni
up,md,low = thirds_pct
if low > (up+md)/2*1.1: rep.append("Long face.")
elif low < (up+md)/2*0.9: rep.append("Short face.")
else: rep.append("Normale.")

st.markdown("- ".join(rep))
st.caption("Guida clinica: percentuali riferite all’altezza/larghezza facciale; validare con esame diretto.")
