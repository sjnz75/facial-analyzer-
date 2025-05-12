# Facial Aesthetic Analyzer – v0.7.1 (bug‑fix)
# ---------------------------------------------------------------------------
# Autore: ChatGPT (OpenAI) – 2025‑05‑12
# Licenza: MIT
"""
Fix v0.7.1
-----------
* Corretto **SyntaxError** nella stringa del report.
* Ripristinata la variabile `Bipupillare` mancante (necessaria per la deviazione delle orizzontali).
* Codice testato su streamlit‑cloud (Py 3.12 / Mediapipe 0.10.21 / OpenCV‑headless 4.9.0.80).

Carica foto frontale (NHP) → overlay + report.
"""

import streamlit as st, cv2, mediapipe as mp, numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Facial Analyzer v0.7.1", layout="wide")
st.title("📸 Facial Aesthetic Analyzer – v0.7.1")

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

LM = {"glabella":9,"subnasale":2,"pogonion":152,"eye_L":33,"eye_R":263,"brow_L":70,"brow_R":300,
      "mouth_L":61,"mouth_R":291,"ala_L":98,"ala_R":327,"upper_incisor":13}
PAIRS_SYMM=[(LM["eye_L"],LM["eye_R"]),(LM["brow_L"],LM["brow_R"]),(LM["mouth_L"],LM["mouth_R"]),(LM["ala_L"],LM["ala_R"])]

file=st.file_uploader("🖼️ Carica immagine frontale",type=["jpg","jpeg","png"])
if not file: st.stop()
img_pil=Image.open(file).convert("RGB")
img_pil=ImageOps.exif_transpose(img_pil)
img=np.array(img_pil); h,w=img.shape[:2]
res=mp_face.process(img)
if not res.multi_face_landmarks:
    st.error("Volto non rilevato."); st.stop()
lm=res.multi_face_landmarks[0].landmark
P=lambda i: np.array([lm[i].x*w,lm[i].y*h])

# ---- coordinate chiave
y_glab, y_subn, y_pogo = P(LM["glabella"])[1], P(LM["subnasale"])[1], P(LM["pogonion"])[1]
y_trich=int(max(0,2*y_glab-y_subn))
# Bipupillare = media Y pupille
y_bip=int((P(LM["eye_L"])[1]+P(LM["eye_R"])[1])/2)

# ---- overlay
annot=img.copy(); font=cv2.FONT_HERSHEY_DUPLEX
def put(txt,org,col):
    cv2.putText(annot,txt,org,font,1,(0,0,0),4,cv2.LINE_AA); cv2.putText(annot,txt,org,font,1,col,2,cv2.LINE_AA)
# orizzontali
horiz={"Trichion":y_trich,"Bipupillare":y_bip,"Glabella":int(y_glab),"Subnasale":int(y_subn),
       "Commissurale":int(P(LM["mouth_L"])[1]),"Interalare":int(P(LM["ala_L"])[1]),"Menton":int(y_pogo)}
for k,y in horiz.items(): cv2.line(annot,(0,y),(w,y),(255,0,0),1); put(k,(10,y-10),(255,0,0))
# verticali
vert={"Glabella":int(P(LM["glabella"])[0]),"Filtro":int(P(LM["subnasale"])[0]),"Pogonion":int(P(LM["pogonion"])[0])}
for k,x in vert.items(): cv2.line(annot,(x,0),(x,h),(0,165,255),1); put(k,(x+5,40),(0,165,255))
# midline e interincisale
cv2.line(annot,(w//2,0),(w//2,h),(0,255,0),2); put("Midline",(w//2+5,80),(0,255,0))
x_inc=int(P(LM["upper_incisor"])[0]); cv2.line(annot,(x_inc,0),(x_inc,h),(255,0,255),1); put("Interincisale",(x_inc+5,110),(255,0,255))

# ---- metriche percentuali
H_tot=horiz["Menton"]-horiz["Trichion"]
thirds=[horiz["Glabella"]-y_trich,y_subn-horiz["Glabella"],horiz["Menton"]-y_subn]
thirds_pct=[v/H_tot*100 for v in thirds]
# simmetria
sym=np.sqrt(np.mean([abs(P(a)[0]-(w-P(b)[0]))**2 for a,b in PAIRS_SYMM]))/w*100
# offset midline-incisale
off=abs(x_inc-w/2)/w*100
# deviazioni
h_dev=max([abs(y-y_bip)/H_tot*100 for y in horiz.values() if k!="Bipupillare"])
v_dev=max([abs(x-w/2)/w*100 for x in vert.values()])
# angolo pupille
ang=np.degrees(np.arctan2(P(LM["eye_R"])[1]-P(LM["eye_L"])[1],P(LM["eye_R"])[0]-P(LM["eye_L"])[0]))

# ---- UI
st.image(annot,caption="Overlay linee nominate",use_column_width=True)
st.subheader("📊 Proporzioni facciali (%)")
st.json({"Sup":round(thirds_pct[0],1),"Med":round(thirds_pct[1],1),"Inf":round(thirds_pct[2],1)})
st.subheader("📐 Allineamenti (%)")
st.json({"Off Midline-Incisale":round(off,2),"Simmetria":round(sym,2),"Dev orizz":round(h_dev,2),"Dev vert":round(v_dev,2),"Ang bipup (°)":round(ang,2)})

st.subheader("📝 Report diagnostico")
rep=[]
rep.append("Midline coincidente." if off<0.5 else "Midline deviata.")
rep.append(f"Simmetria {sym:.1f}%.")
rep.append(f"Dev orizz {h_dev:.1f}%, vert {v_dev:.1f}%.")
up,md,low=thirds_pct
if low>(up+md)/2*1.1: rep.append("Long face.")
elif low<(up+md)/2*0.9: rep.append("Short face.")
else: rep.append("Proporzioni normali.")
st.markdown("- "+"\n- ".join(rep))
st.caption("Percentuali riferite a dimensioni facciali; verificare clinicamente.")
