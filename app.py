# app.py — Facial Aesthetic Analyzer **v1.4.1** (bug‑fix)
# ===============================================================
"""Solo fix variabile
• `img_rgba` ora è definito correttamente (prima c’era un refuso `ing_rgba`)  
• Il canvas usa `img_rgba` come `background_image`
"""

import streamlit as st, numpy as np, mediapipe as mp
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Facial Analyzer v1.4.1", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v1.4.1 (interattivo)")

st.markdown("""1. Carica la foto frontale (NHP).  
2. Trascina le linee dove servono.  
3. Premi **Ricalcola diagnosi**.""")

MP_FACE = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
LM={"glabella":9,"subnasale":2,"pogonion":152,"eye_L":33,"eye_R":263,"mouth_L":61,"mouth_R":291,"ala_L":98,"ala_R":327,"upper_incisor":13}
PAIRS=[(LM["eye_L"],LM["eye_R"]),(LM["mouth_L"],LM["mouth_R"]),(LM["ala_L"],LM["ala_R"])]
RANGE={"terzi":"33 ± 3 %","midline":"< 0.4 %","simmetria":"< 3 %"}

file=st.file_uploader("🖼️ Carica immagine",type=["jpg","jpeg","png"])
if not file: st.stop()
img_pil=Image.open(file).convert("RGB")
img_pil=ImageOps.exif_transpose(img_pil)
if img_pil.width>800:
    r=800/img_pil.width
    img_pil=img_pil.resize((800,int(img_pil.height*r)))

# Fix: variabile corretta
img_rgba=img_pil.convert("RGBA")
img=np.array(img_rgba); H,W=img.shape[:2]

res=MP_FACE.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato."); st.stop()
lm=res.multi_face_landmarks[0].landmark
P=lambda i: np.array([lm[i].x*W,lm[i].y*H])

# linee base
yG,yS,yP=P(LM["glabella"])[1],P(LM["subnasale"])[1],P(LM["pogonion"])[1]
yT=max(0,int(2*yG-yS))
yB=int((P(LM["eye_L"])[1]+P(LM["eye_R"])[1])/2)

init={
 "Trichion":((0,yT),(W,yT),"#1E90FF"),
 "Glabella":((0,int(yG)),(W,int(yG)),"#1E90FF"),
 "Bipupillare":((0,yB),(W,yB),"#1E90FF"),
 "Subnasale":((0,int(yS)),(W,int(yS)),"#1E90FF"),
 "Interalare":((0,int(P(LM["ala_L"])[1])),(W,int(P(LM["ala_L"])[1])),"#1E90FF"),
 "Menton":((0,int(yP)),(W,int(yP)),"#1E90FF"),
 "Midline":((W//2,0),(W//2,H),"#32CD32"),
 "Interincisale":((int(P(LM["upper_incisor"])[0]),0),(int(P(LM["upper_incisor"])[0]),H),"#FF1493"),
}
json_init={"version":"4.4.0","objects":[{"type":"line","name":k,"stroke":c,"strokeWidth":2,"x1":x1,"y1":y1,"x2":x2,"y2":y2} for k,((x1,y1),(x2,y2),c) in init.items()]}

canvas=st_canvas(background_image=img_rgba, initial_drawing=json_init, update_streamlit=True, height=H, width=W, drawing_mode="transform", key="canvas")
objs={o.get("name"):o for o in canvas.json_data.get("objects",[]) if o.get("name")} if canvas.json_data else {}
cy=lambda n:(objs[n]["y1"]+objs[n]["y2"])/2
cx=lambda n:(objs[n]["x1"]+objs[n]["x2"])/2

if st.button("Ricalcola diagnosi"):
    if not all(k in objs for k in ["Trichion","Glabella","Subnasale","Menton","Midline","Interincisale"]):
        st.error("Linee mancanti: ricarica pagina."); st.stop()
    yT,yG,yS,yM=[cy(k) for k in ["Trichion","Glabella","Subnasale","Menton"]]
    thirds=((yG-yT),(yS-yG),(yM-yS)); thirds=np.array(thirds)/(yM-yT)*100
    off=abs(cx("Interincisale")-cx("Midline"))/W*100
    sym=np.sqrt(np.mean([abs(P(a)[0]-(W-P(b)[0]))**2 for a,b in PAIRS]))/W*100

    st.json({"Sup":round(float(thirds[0]),1),"Med":round(float(thirds[1]),1),"Inf":round(float(thirds[2]),1),"Midline":round(off,2),"Simmetria":round(sym,2)})
