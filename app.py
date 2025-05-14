import streamlit as st
import numpy as np
import cv2
from PIL import Image              #  ← spostalo qui, PRIMA di usare Image
from streamlit_drawable_canvas import st_canvas
import base64, io                  #  ← utility per data-URL
def resize_for_canvas(img: Image.Image, max_w: int = 700) -> Image.Image:
    """Riduce l’immagine preservando il rapporto se supera max_w pixel."""
    if img.width > max_w:
        ratio = max_w / img.width
        new_size = (max_w, int(img.height * ratio))
        return img.resize(new_size)
    return img
def pil_to_data_url(img: Image.Image) -> str:
    """Converte una PIL.Image in data-URL base64 (PNG) per st_canvas."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{data}"
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from utils.geometry import (
    line_angle_deg,
    line_deviation_mm,
    segment_length,
    compute_vertical_thirds
)
from utils.diagnostics import (
    diagnose_median_line,
    diagnose_angle_line,
    diagnose_thirds
)

st.set_page_config(page_title="Face Aesthetics Analyzer", layout="wide")

st.title("Face Aesthetics Analyzer 🖼️➡️📊")

# Caricamento immagine
uploaded_file = st.file_uploader("Carica un'immagine frontale del volto", type=["png", "jpg", "jpeg"])
if not uploaded_file:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
width, height = image.size

# Definizione dei landmark da cliccare in ordine
landmark_labels = [
    "Glabella (Gl)", "Subnasale (Sn)", "Pogonion (Pg')",                # Linea mediana
    "Centro Pupilla Destra", "Centro Pupilla Sinistra",                # Interpupillare
    "Commissura Destra", "Commissura Sinistra",                        # Commissurale
    "Ala Nasale Destra", "Ala Nasale Sinistra",                        # Interalare
    "Trichion (Tri)", "Glabella (Gl)", "Subnasale (Sn)", "Menton (Me')"  # Terzi verticali
]

st.markdown("**Istruzioni:** clicca i punti nell'ordine mostrato e premi 'Termina selezione'.")
canvas_result = st_canvas(
    fill_color="",
    stroke_width=3,
    stroke_color="red",
image = resize_for_canvas(image),
    update_streamlit=True,
    height=height,
    width=width,
    drawing_mode="point",

    key="canvas"
)

if st.button("Termina selezione"):
    objs = canvas_result.json_data["objects"]
    if len(objs) != len(landmark_labels):
        st.error(f"Devi selezionare {len(landmark_labels)} punti, ne hai selezionati {len(objs)}.")
        st.stop()

    # Estrazione coordinate
    pts = {label: (obj["left"], obj["top"]) for label, obj in zip(landmark_labels, objs)}

    # Calcoli geometrici
    # Linea mediana
    med_angle = line_angle_deg(pts["Glabella (Gl)"], pts["Pogonion (Pg')"])
    med_dev = line_deviation_mm(pts["Glabella (Gl)"], pts["Subnasale (Sn)"], pts["Pogonion (Pg')"])
    # Interpupillare
    ip_angle = line_angle_deg(pts["Centro Pupilla Destra"], pts["Centro Pupilla Sinistra"], horizontal=True)
    # Commissurale
    com_angle = line_angle_deg(pts["Commissura Destra"], pts["Commissura Sinistra"], horizontal=True)
    # Interalare: differenza y delle ali
    ala_diff = abs(pts["Ala Nasale Destra"][1] - pts["Ala Nasale Sinistra"][1])
    ala_angle = line_angle_deg(pts["Ala Nasale Destra"], pts["Ala Nasale Sinistra"], horizontal=True)
    # Terzi verticali
    thirds = compute_vertical_thirds(
        pts["Trichion (Tri)"], pts["Glabella (Gl)"], pts["Subnasale (Sn)"], pts["Menton (Me')"]
    )

    # Diagnosi
    diag_med = diagnose_median_line(med_dev)
    diag_ip = diagnose_angle_line(ip_angle, line_type="interpupillare")
    diag_com = diagnose_angle_line(com_angle, line_type="commissurale")
    diag_int = "Asimmetria medio-facciale" if ala_diff > 2 else "Normale"
    diag_thirds = diagnose_thirds(thirds)

    # Annotazione immagine
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Disegna linee e angoli (esempio per interpupillare)
    cv2.line(img_cv, tuple(map(int, pts["Centro Pupilla Destra"])), tuple(map(int, pts["Centro Pupilla Sinistra"])), (0,255,0), 2)
    cv2.putText(img_cv, f"IP: {ip_angle:.1f}°", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    st.image(annotated, caption="Annotazione Clinica", use_column_width=True)

    # Tabella riassuntiva
    import pandas as pd
    df = pd.DataFrame([
        ["Linea Mediana", f"{med_dev:.1f} mm / {med_angle:.1f}°", diag_med],
        ["Interpupillare", f"{ip_angle:.1f}°", diag_ip],
        ["Commissurale", f"{com_angle:.1f}°", diag_com],
        ["Interalare", f"{ala_diff:.1f} mm", diag_int],
        ["Terzi Verticali", "; ".join([f"{v:.1f}%" for v in thirds]), "; ".join(diag_thirds)]
    ], columns=["Linea", "Misura", "Diagnosi"] )
    st.table(df)

    # Commenti clinici aggiuntivi
    st.markdown("**Commenti Clinici:**")
    st.markdown(f"- Linea mediana: {diag_med}")
    st.markdown(f"- Terzi verticali: {', '.join(diag_thirds)}")
