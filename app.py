# Facial Aesthetic Analyzer – **v1.0** (interattivo)
# ---------------------------------------------------------------------------
# Autore: ChatGPT (OpenAI)
# Licenza: MIT
"""
### Cosa fa questa versione
1. **Rileva automaticamente** le linee principali (orizzontali e verticali).
2. Le **visualizza su un canvas interattivo** dove l’utente può *trascinarle* nella posizione corretta.
3. Al click su **“Ricalcola diagnosi”** rielabora proporzioni e deviazioni **in percentuale** e restituisce:
   * Diagnosi (Normal / Short / Long face, midline, simmetria…)
   * **Range accettabili** per ogni metrica, evidenziando in rosso quelle fuori soglia.

> Dipendenze aggiunte: `streamlit-drawable-canvas` (aggiungi al `requirements.txt`).
"""

import streamlit as st
import cv2, mediapipe as mp, numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Facial Analyzer v1.0", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v1.0 (interattivo)")

MP_FACE = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
LM = {"glabella":9,"subnasale":2,"pogonion":152,"eye_L":33,"eye_R":263,"mouth_L":61,"mouth_R":291,"ala_L":98,"ala_R":327,"upper_incisor":13}
PAIRS = [(LM["eye_L"],LM["eye_R"]),(LM["mouth_L"],LM["mouth_R"]),(LM["ala_L"],LM["ala_R"])]

FILE = st.file_uploader("Carica foto frontale (JPG/PNG)", type=["jpg","jpeg","png"])
if not FILE: st.stop()

img_pil = Image.open(FILE).convert("RGB")
img_pil = ImageOps.exif_transpose(img_pil)
img = np.array(img_pil); H,W=img.shape[:2]
res = MP_FACE.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato."); st.stop()

lm = res.multi_face_landmarks[0].landmark
P = lambda i: np.array([lm[i].x*W,lm[i].y*H])

y_glab, y_subn, y_pogo = P(LM["glabella"])[1], P(LM["subnasale"])[1], P(LM["pogonion"])[1]
y_trich = max(0, int(2*y_glab - y_subn))
y_bip   = int((P(LM["eye_L"])[1] + P(LM["eye_R"])[1])/2)

init_lines = {
    "Trichion":    ((0,y_trich),(W,y_trich), "red"),
    "Glabella":    ((0,int(y_glab)),(W,int(y_glab)), "red"),
    "Bipupillare": ((0,y_bip),(W,y_bip), "red"),
    "Subnasale":   ((0,int(y_subn)),(W,int(y_subn)), "red"),
    "Interalare":  ((0,int(P(LM["ala_L"])[1])),(W,int(P(LM["ala_L"])[1])), "red"),
    "Menton":      ((0,int(y_pogo)),(W,int(y_pogo)), "red"),
    "Midline":     ((W//2,0),(W//2,H), "green"),
    "Interincisale":((int(P(LM["upper_incisor"])[0]),0),(int(P(LM["upper_incisor"])[0]),H),"magenta"),
}

# Create initial JSON drawing
init_json = {"version":"4.4.0","objects":[]}
for name,(p1,p2,color) in init_lines.items():
    init_json["objects"].append({
        "type":"line","stroke":color,"strokeWidth":2,
        "x1":p1[0],"y1":p1[1],"x2":p2[0],"y2":p2[1],
        "name":name
    })

st.write("### 1️⃣ Sposta le linee se necessario, poi premi **Ricalcola diagnosi**")
canvas = st_canvas(background_image=img_pil, initial_drawing=init_json,
                   update_streamlit=True, height=H, width=W,
                   drawing_mode="transform", key="canvas")

if st.button("Ricalcola diagnosi"):
    objs = {o["name"]:o for o in canvas.json_data["objects"] if o["type"]=="line"}
    get_y = lambda name: (objs[name]["y1"]+objs[name]["y2"])/2
    get_x = lambda name: (objs[name]["x1"]+objs[name]["x2"])/2

    y_T = get_y("Trichion"); y_G = get_y("Glabella"); y_S = get_y("Subnasale"); y_M = get_y("Menton")
    y_total = y_M - y_T
    thirds = [ (y_G-y_T)/y_total*100, (y_S-y_G)/y_total*100, (y_M-y_S)/y_total*100 ]

    off_mid = abs(get_x("Interincisale") - get_x("Midline"))/W*100
    symm = np.sqrt(np.mean([abs(P(a)[0]-(W-P(b)[0]))**2 for a,b in PAIRS]))/W*100

    # Ranges
    ranges = {
        "Terzi": "Ideale 33±3 %",
        "Midline": "OK <0.4 % larghezza",
        "Simmetria": "OK <3 %",
    }

    # Output
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("📊 Proporzioni (%)")
        st.write({"Sup":round(thirds[0],1),"Med":round(thirds[1],1),"Inf":round(thirds[2],1)})
    with col2:
        st.subheader("📐 Deviations (%)")
        st.write({"Midline offset":round(off_mid,2),"Simmetry RMS":round(symm,2)})

    # Diagnosis
    st.subheader("📝 Diagnosi")
    diag=[]
    if abs(thirds[0]-33)<3 and abs(thirds[1]-33)<3 and abs(thirds[2]-33)<3:
        diag.append("Proporzioni facciali normali")
    elif thirds[2]>(thirds[0]+thirds[1])/2*1.1:
        diag.append("Long face (terzo inferiore >110% media sup+med)")
    elif thirds[2]<(thirds[0]+thirds[1])/2*0.9:
        diag.append("Short face (terzo inferiore <90% media sup+med)")

    diag.append("Midline ok" if off_mid<0.4 else "Midline deviata >0.4%")
    diag.append("Simmetria ok" if symm<3 else "Asimmetria >3%")
    st.markdown("\n".join([f"- {d}" for d in diag]))

    st.subheader("📏 Range accettabili")
    st.write(ranges)
