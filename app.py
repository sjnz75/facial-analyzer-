# app.py — Facial Aesthetic Analyzer **v1.3** (bug-free)
# ===========================================================
# Autore: ChatGPT (OpenAI) – 2025-05-12
# Licenza: MIT
"""
Dipendenze (requirements.txt) – mantenere esattamente queste righe:
    streamlit==1.34.0
    mediapipe==0.10.21
    opencv-contrib-python-headless==4.11.0.86
    numpy
    pillow==10.3.0
    streamlit-drawable-canvas==0.9.3
"""

import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ────────────────── Config pagina ──────────────────
st.set_page_config(page_title="Facial Analyzer v1.3", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v1.3 (interattivo)")

st.markdown("""
**Workflow**  
1. Carica la foto frontale (NHP).  
2. Trascina le linee dove servono.  
3. Premi **Ricalcola diagnosi** per aggiornare valori e report.
""")

# ────────────────── Costanti ──────────────────
MP_FACE = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
LM = {"glabella":9,"subnasale":2,"pogonion":152,"eye_L":33,"eye_R":263,"mouth_L":61,"mouth_R":291,"ala_L":98,"ala_R":327,"upper_incisor":13}
PAIRS_SYMM=[(LM["eye_L"],LM["eye_R"]),(LM["mouth_L"],LM["mouth_R"]),(LM["ala_L"],LM["ala_R"])]
RANGE={"terzi":"33 ± 3 %","midline":"< 0.4 %","simmetria":"< 3 %"}

# ────────────────── Upload & resize ──────────────────
file = st.file_uploader("🖼️ Carica immagine", type=["jpg","jpeg","png"])
if not file:
    st.stop()
img_pil = Image.open(file).convert("RGB")
img_pil = ImageOps.exif_transpose(img_pil)
MAX_W = 800
if img_pil.width > MAX_W:
    ratio = MAX_W / img_pil.width
    img_pil = img_pil.resize((MAX_W, int(img_pil.height*ratio)))
st.image(img_pil, caption="Anteprima foto", use_column_width=False)

img = np.array(img_pil); H,W = img.shape[:2]

# ────────────────── Landmark detection ──────────────────
res = MP_FACE.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato."); st.stop()
lm=res.multi_face_landmarks[0].landmark
P=lambda i: np.array([lm[i].x*W, lm[i].y*H])

# prime stime linee
y_G,y_S,y_P = P(LM["glabella"])[1],P(LM["subnasale"])[1],P(LM["pogonion"])[1]
y_T = max(0,int(2*y_G - y_S))
 y_B = int((P(LM["eye_L"])[1]+P(LM["eye_R"])[1])/2)

init = {
 "Trichion":      ((0,y_T),(W,y_T),"#1E90FF"),
 "Glabella":      ((0,int(y_G)),(W,int(y_G)),"#1E90FF"),
 "Bipupillare":   ((0,y_B),(W,y_B),"#1E90FF"),
 "Subnasale":     ((0,int(y_S)),(W,int(y_S)),"#1E90FF"),
 "Interalare":    ((0,int(P(LM["ala_L"])[1])),(W,int(P(LM["ala_L"])[1])),"#1E90FF"),
 "Menton":        ((0,int(y_P)),(W,int(y_P)),"#1E90FF"),
 "Midline":       ((W//2,0),(W//2,H),"#32CD32"),
 "Interincisale": ((int(P(LM["upper_incisor"])[0]),0),(int(P(LM["upper_incisor"])[0]),H),"#FF1493")
}
json_init={"version":"4.4.0","objects":[{"type":"line","name":k,"stroke":c,"strokeWidth":2,"x1":x1,"y1":y1,"x2":x2,"y2":y2} for k,((x1,y1),(x2,y2),c) in init.items()]}

st.write("### 1️⃣ Sposta le linee, poi clicca **Ricalcola diagnosi**")
canvas=st_canvas(background_image=img_pil,initial_drawing=json_init,update_streamlit=True,height=H,width=W,drawing_mode="transform",key="canvas")

# helper
objs={}
if canvas.json_data and "objects" in canvas.json_data:
    for o in canvas.json_data["objects"]:
        name=o.get("name")
        if name: objs[name]=o
center_y=lambda n:(objs[n]["y1"]+objs[n]["y2"])/2 if n in objs else None
center_x=lambda n:(objs[n]["x1"]+objs[n]["x2"])/2 if n in objs else None

if st.button("Ricalcola diagnosi"):
    needed=["Trichion","Glabella","Subnasale","Menton","Midline","Interincisale"]
    if not all(k in objs for k in needed):
        st.error("Linee mancanti: ricarica la pagina."); st.stop()

    y_T,y_G,y_S,y_M=[center_y(k) for k in ["Trichion","Glabella","Subnasale","Menton"]]
    H_tot=y_M-y_T
    thirds=((y_G-y_T)/H_tot*100,(y_S-y_G)/H_tot*100,(y_M-y_S)/H_tot*100)

    off_pct=abs(center_x("Interincisale")-center_x("Midline"))/W*100
    sym_pct=np.sqrt(np.mean([abs(P(a)[0]-(W-P(b)[0]))**2 for a,b in PAIRS_SYMM]))/W*100

    col1,col2=st.columns(2)
    with col1:
        st.subheader("📊 Proporzioni (%)")
        st.json({"Sup":round(thirds[0],1),"Med":round(thirds[1],1),"Inf":round(thirds[2],1)})
    with col2:
        st.subheader("📐 Deviazioni (%)")
        st.json({"Midline":round(off_pct,2),"Simmetria":round(sym_pct,2)})

    diag=[]
    if all(abs(t-33)<3 for t in thirds):
        diag.append("Proporzioni: normali (33 ± 3 %).")
    elif thirds[2]>(thirds[0]+thirds[1])/2*1.1:
        diag.append("Long face: terzo inferiore aumentato.")
    elif thirds[2]<(thirds[0]+thirds[1])/2*0.9:
        diag.append("Short face: terzo inferiore ridotto.")
    else:
        diag.append("Proporzioni fuori range ideale.")
    diag.append("Midline ok" if off_pct<0.4 else "Midline deviata (>0.4 %).")
    diag.append("Simmetria ok" if sym_pct<3 else "Asimmetria (>3 %).")

    st.subheader("📝 Diagnosi")
    st.markdown("\n".join(["- "+d for d in diag]))

    st.subheader("🏷️ Range clinici")
    st.json(RANGE)

    st.caption("Valori percentuali da foto 2D; confermare con valutazione clinica.")
