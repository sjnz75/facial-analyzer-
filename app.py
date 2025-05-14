import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io, base64      # ti servono ancora solo se in futuro farai export

# ---------- utility ----------
def resize_for_canvas(img: Image.Image, max_w: int = 700) -> Image.Image:
    """Riduce l’immagine preservando il rapporto se supera max_w pixel."""
    if img.width > max_w:
        ratio = max_w / img.width
        new_size = (max_w, int(img.height * ratio))
        return img.resize(new_size)
    return img

# ---------- import moduli locali ----------
from utils.geometry import (
    line_angle_deg,
    line_deviation_mm,
    segment_length,
    compute_vertical_thirds,
)
from utils.diagnostics import (
    diagnose_median_line,
    diagnose_angle_line,
    diagnose_thirds,
)

# ---------- pagina ----------
st.set_page_config(page_title="Face Aesthetics Analyzer", layout="wide")
st.title("Face Aesthetics Analyzer 🖼️ ➡️ 📊")

# ---------- caricamento ----------
uploaded_file = st.file_uploader(
    "Carica un'immagine frontale del volto", type=["png", "jpg", "jpeg"]
)
if not uploaded_file:
    st.stop()

# lettura + resize
image = Image.open(uploaded_file).convert("RGB")
image = resize_for_canvas(image)     # max 700 px di larghezza

width, height = image.size



# ---------- canvas ----------
landmark_labels = [
    "Glabella (Gl)", "Subnasale (Sn)", "Pogonion (Pg')",          # linea mediana
    "Centro Pupilla Destra", "Centro Pupilla Sinistra",          # interpupillare
    "Commissura Destra", "Commissura Sinistra",                  # commissurale
    "Ala Nasale Destra", "Ala Nasale Sinistra",                  # interalare
    "Trichion (Tri)", "Glabella (Gl)", "Subnasale (Sn)", "Menton (Me')"  # terzi
]

st.markdown("**Istruzioni:** clicca i punti nell'ordine e premi 'Termina selezione'.")

canvas_result = st_canvas(
    fill_color="",
    stroke_width=3,
    stroke_color="red",
    background_image=image,      # PIL
    background_color="#00000000",
    update_streamlit=True,
    height=image.height,
    width=image.width,
    drawing_mode="point",
    point_display_radius=6,
    display_toolbar=False,
    key="canvas"
)
st.write("DEBUG → size:", image.width, "x", image.height)
st.image(image.resize((150, int(150 * image.height / image.width))),
         caption="DEBUG thumbnail (se la vedi, l’immagine è OK)")
# ---------- elaborazione punti ----------
if st.button("Termina selezione"):
    objs = canvas_result.json_data["objects"]
    if len(objs) != len(landmark_labels):
        st.error(f"Devi selezionare {len(landmark_labels)} punti, ne hai selezionati {len(objs)}.")
        st.stop()

    pts = {label: (obj["left"], obj["top"]) for label, obj in zip(landmark_labels, objs)}

    # --- calcoli geometrici ---
    med_angle = line_angle_deg(pts["Glabella (Gl)"], pts["Pogonion (Pg')"])
    med_dev   = line_deviation_mm(pts["Glabella (Gl)"], pts["Subnasale (Sn)"], pts["Pogonion (Pg')"])
    ip_angle  = line_angle_deg(pts["Centro Pupilla Destra"], pts["Centro Pupilla Sinistra"], horizontal=True)
    com_angle = line_angle_deg(pts["Commissura Destra"],  pts["Commissura Sinistra"],  horizontal=True)
    ala_diff  = abs(pts["Ala Nasale Destra"][1] - pts["Ala Nasale Sinistra"][1])
    thirds    = compute_vertical_thirds(
        pts["Trichion (Tri)"], pts["Glabella (Gl)"], pts["Subnasale (Sn)"], pts["Menton (Me')"]
    )

    # --- diagnosi ---
    diag_med   = diagnose_median_line(med_dev)
    diag_ip    = diagnose_angle_line(ip_angle,  "interpupillare")
    diag_com   = diagnose_angle_line(com_angle, "commissurale")
    diag_int   = "Asimmetria medio-facciale" if ala_diff > 2 else "Normale"
    diag_thirds = diagnose_thirds(thirds)

    # --- annotazione immagine ---
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.line(img_cv, tuple(map(int, pts["Centro Pupilla Destra"])),
                     tuple(map(int, pts["Centro Pupilla Sinistra"])), (0,255,0), 2)
    cv2.putText(img_cv, f"IP: {ip_angle:.1f}°", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    annotated = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    st.image(annotated, caption="Annotazione clinica", use_column_width=True)

    # --- tabella ---
    import pandas as pd
    df = pd.DataFrame([
        ["Linea Mediana",   f"{med_dev:.1f} mm / {med_angle:.1f}°", diag_med],
        ["Interpupillare",  f"{ip_angle:.1f}°",                    diag_ip],
        ["Commissurale",    f"{com_angle:.1f}°",                   diag_com],
        ["Interalare",      f"{ala_diff:.1f} mm",                  diag_int],
        ["Terzi Verticali", "; ".join(f"{v:.1f}%" for v in thirds), "; ".join(diag_thirds)],
    ], columns=["Linea", "Misura", "Diagnosi"])
    st.table(df)

    # --- commenti ---
    st.markdown("**Commenti clinici sintetici**")
    st.markdown(f"- Linea mediana: {diag_med}")
    st.markdown(f"- Terzi verticali: {', '.join(diag_thirds)}")
