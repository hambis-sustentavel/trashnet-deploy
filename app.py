# app_garbage_detector.py
# Detector de Resíduos com YOLOv5

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
import tempfile
from pathlib import Path
import numpy as np

# Configurações
MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
IMG_SIZE = 640
DEFAULT_CONF = 0.25

# Cores para as classes
COLORS = [
    "#FF595E", "#FFCA3A", "#8AC926", 
    "#1982C4", "#6A4C93", "#FF924C"
]

def _hex2rgb(hexcode: str) -> tuple[int, int, int]:
    hexcode = hexcode.lstrip("#")
    return tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def load_model():
    """Carrega o modelo YOLOv5 usando ultralytics."""
    try:
        from ultralytics import YOLO
        
        # Usar modelo pré-treinado do YOLOv5 (genérico)
        # Para simplificar, vamos usar o modelo padrão que detecta objetos gerais
        model = YOLO('yolov5s.pt')
        return model
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None

def detect(model, image, conf_threshold):
    """Executa detecção de objetos na imagem."""
    if model is None:
        return []
    
    try:
        # Executa detecção
        results = model(image, conf=conf_threshold, imgsz=IMG_SIZE, verbose=False)
        
        # Processa resultados
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Coordenadas da caixa
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    # Confiança e classe
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
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
st.set_page_config(page_title="Detector de Objetos", layout="wide")
st.title("� Detector de Objetos (YOLOv5)")

# Controle de confiança
conf_slider = st.sidebar.slider(
    "Confiança mínima (%)", 5, 90, int(DEFAULT_CONF * 100), 1
) / 100

# Carrega modelo
model = load_model()

# Classes do YOLO padrão (simplificadas para objetos comuns)
CLASSES = {
    0: "pessoa", 39: "garrafa", 41: "xícara", 42: "garfo", 43: "faca", 44: "colher", 45: "tigela",
    46: "banana", 47: "maçã", 48: "sanduíche", 49: "laranja", 50: "brócolis", 51: "cenoura",
    52: "cachorro-quente", 53: "pizza", 54: "donut", 55: "bolo", 56: "cadeira", 57: "sofá",
    58: "planta", 59: "cama", 60: "mesa", 61: "vaso sanitário", 62: "tv", 63: "laptop",
    64: "mouse", 65: "controle", 66: "teclado", 67: "celular", 68: "microondas", 69: "forno",
    70: "torradeira", 71: "pia", 72: "geladeira", 73: "livro", 74: "relógio", 75: "vaso",
    76: "tesoura", 77: "urso de pelúcia", 78: "secador", 79: "escova de dente"
}

if model:
    st.sidebar.markdown("**Detecta objetos comuns do dia a dia**")
else:
    st.sidebar.error("Modelo não carregado")

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
        st.warning("Nenhum objeto detectado acima do limiar escolhido.")
        with st.expander("💡 Dicas para melhores resultados"):
            st.write(
                "- Diminua o slider de confiança\n"
                "- Certifique-se de que há objetos visíveis na imagem\n"
                "- Use boa iluminação\n"
                "- O modelo detecta objetos comuns como pessoas, garrafas, móveis, etc."
            )
    else:
        img_bb = draw_boxes(img.copy(), detections, CLASSES)
        st.image(img_bb, caption="Detecções", use_container_width=True)

        st.subheader("Detalhes")
        for (_, _, _, _), cls_id, conf in detections:
            class_name = CLASSES.get(cls_id, f"Classe {cls_id}")
            st.write(f"- **{class_name}** — {conf:.1%}")

elif not model:
    st.error("❌ Falha ao carregar o modelo. Tente recarregar a página.")
