# app_garbage_detector.py
# Requisitos:
#   pip install streamlit torch pillow ultralytics

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import requests
import tempfile
from pathlib import Path

# ───────────────────────── Configurações ─────────────────────────
MODEL_URL = "https://huggingface.co/keremberke/yolov5s-garbage/resolve/main/best.pt"
IMG_SIZE = 640
DEFAULT_CONF = 0.25

# Paleta de cores por classe
COLORS = [
    "#FF595E", "#FFCA3A", "#8AC926",
    "#1982C4", "#6A4C93", "#FF924C"
]

def _hex2rgb(hexcode: str) -> tuple[int, int, int]:
    hexcode = hexcode.lstrip("#")
    return tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))

@st.cache_resource(show_spinner="🔄 Baixando modelo...")
def load_model():
    """Baixa e carrega o modelo YOLOv5 customizado."""
    try:
        # Baixa o modelo
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        
        # Salva temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_file.write(response.content)
            model_path = tmp_file.name
        
        # Carrega com ultralytics
        model = YOLO(model_path)
        
        # Remove arquivo temporário
        Path(model_path).unlink()
        
        return model
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None

def detect(model, image, conf_threshold):
    """Executa detecção de objetos na imagem."""
    if model is None:
        return []
    
    try:
        results = model(image, conf=conf_threshold, imgsz=IMG_SIZE)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Coordenadas da caixa
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Confiança e classe
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append(((x1, y1, x2, y2), cls, conf))
        
        return detections
        
    except Exception as e:
        st.error(f"❌ Erro na detecção: {e}")
        return []

def draw_boxes(img: Image.Image, detections, class_names):
    """Desenha bounding boxes na imagem."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), cls_id, conf in detections:
        color = _hex2rgb(COLORS[cls_id % len(COLORS)])
        label = f"{class_names[cls_id]} {conf:.0%}"

        # Caixa
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Texto
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        draw.text((x1, y1 - th), label, fill="white", font=font)

    return img


# ───────────────────────────── UI ────────────────────────────────
st.set_page_config(page_title="Detector de Resíduos", layout="wide")
st.title("🗑️ Detector de Resíduos Hambis (YOLOv5s-Garbage)")

# Controle de confiança
conf_slider = st.sidebar.slider(
    "Confiança mínima (%)", 5, 90, int(DEFAULT_CONF * 100), 1
) / 100

# Carrega modelo
model = load_model()

# Classes do modelo (hardcoded para este modelo específico)
CLASSES = {
    0: "biodegradable",
    1: "cardboard", 
    2: "glass",
    3: "metal",
    4: "paper",
    5: "plastic"
}

st.sidebar.markdown(f"**Classes:** {', '.join(CLASSES.values())}")

# Fonte da imagem
source = st.radio("Escolha a fonte da imagem:", ["📷 Câmera", "🖼️ Upload"])
file = (
    st.camera_input("Tire uma foto…")
    if source == "📷 Câmera"
    else st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])
)

if file and model:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Imagem original", use_container_width=True)

    with st.spinner("Detectando…"):
        detections = detect(model, img, conf_slider)

    if not detections:
        st.warning("Nenhum resíduo detectado acima do limiar escolhido.")
        with st.expander("💡 Dicas para melhores resultados"):
            st.write(
                "- Diminua o slider de confiança\n"
                "- Certifique-se de que o objeto aparece inteiro na imagem\n"
                "- Use boa iluminação\n"
                "- Tipos suportados: biodegradável, papelão, vidro, metal, papel, plástico"
            )
    else:
        img_bb = draw_boxes(img.copy(), detections, CLASSES)
        st.image(img_bb, caption="Detecções", use_container_width=True)

        st.subheader("Detalhes")
        for (_, _, _, _), cls_id, conf in detections:
            st.write(f"- **{CLASSES[cls_id]}** — {conf:.1%}")

elif not model:
    st.error("❌ Falha ao carregar o modelo. Tente recarregar a página.")
