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
    """Executa detecção de objetos na imagem, filtrando apenas recicláveis."""
    if model is None:
        return []
    
    try:
        # Executa detecção
        results = model(image, conf=conf_threshold, imgsz=IMG_SIZE, verbose=False)
        
        # Processa resultados, filtrando apenas itens recicláveis
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
                    
                    # Filtrar apenas classes recicláveis
                    if cls in RECYCLABLE_CLASSES:
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
st.set_page_config(page_title="Detector de Recicláveis", layout="wide")
st.title("♻️ Detector de Resíduos Recicláveis")
st.markdown("**Identifica materiais recicláveis como garrafas, latas, copos e utensílios**")

# Controle de confiança
conf_slider = st.sidebar.slider(
    "Confiança mínima (%)", 5, 90, int(DEFAULT_CONF * 100), 1
) / 100

# Carrega modelo
model = load_model()

# Classes do YOLO filtradas para itens recicláveis
RECYCLABLE_CLASSES = {
    # Plásticos
    39: "🥤 Garrafa (Plástico/Vidro)",
    41: "☕ Xícara/Copo",
    44: "🥄 Colher (Plástico/Metal)", 
    45: "🍽️ Tigela/Prato",
    
    # Metais  
    42: "🍴 Garfo (Metal)",
    43: "🔪 Faca (Metal)",
    
    # Eletrônicos
    63: "💻 Laptop",
    64: "🖱️ Mouse",
    67: "📱 Celular", 
    68: "📺 Microondas",
    69: "🔥 Forno",
    70: "🍞 Torradeira",
    72: "❄️ Geladeira",
    
    # Outros recicláveis
    73: "📚 Livro (Papel)",
    74: "⏰ Relógio",
    75: "🏺 Vaso",
    76: "✂️ Tesoura",
    78: "💨 Secador"
}

# Categorias de reciclagem
RECYCLE_CATEGORIES = {
    # Plásticos
    39: "🟡 PLÁSTICO", 41: "🟡 PLÁSTICO", 44: "🟡 PLÁSTICO", 45: "🟡 PLÁSTICO",
    # Metais
    42: "🔵 METAL", 43: "🔵 METAL", 74: "🔵 METAL", 76: "🔵 METAL",
    # Eletrônicos
    63: "🟣 ELETRÔNICO", 64: "🟣 ELETRÔNICO", 67: "🟣 ELETRÔNICO", 
    68: "🟣 ELETRÔNICO", 69: "🟣 ELETRÔNICO", 70: "🟣 ELETRÔNICO", 72: "🟣 ELETRÔNICO", 78: "🟣 ELETRÔNICO",
    # Outros
    73: "🟢 PAPEL", 75: "🟡 PLÁSTICO"
}

if model:
    st.sidebar.markdown("**🔍 Tipos detectados:**")
    st.sidebar.markdown("🟡 **Plásticos** - Garrafas, copos, utensílios")
    st.sidebar.markdown("🔵 **Metais** - Talheres, relógios, tesouras")  
    st.sidebar.markdown("🟣 **Eletrônicos** - Celular, laptop, eletrodomésticos")
    st.sidebar.markdown("🟢 **Papel** - Livros, documentos")
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
        st.warning("Nenhum material reciclável detectado acima do limiar escolhido.")
        with st.expander("💡 Dicas para melhores resultados"):
            st.write(
                "- Diminua o slider de confiança para 5-15%\n"
                "- Certifique-se de que há materiais recicláveis visíveis\n"
                "- Use boa iluminação\n"
                "- **Materiais detectados:** garrafas, copos, talheres, eletrônicos, livros\n"
                "- Aproxime-se dos objetos para melhor detecção"
            )
    else:
        img_bb = draw_boxes(img.copy(), detections, RECYCLABLE_CLASSES)
        st.image(img_bb, caption="Detecções", use_container_width=True)

        st.subheader("📊 Detalhes das Detecções")
        
        # Contador por categoria
        categories_count = {}
        for (_, _, _, _), cls_id, conf in detections:
            category = RECYCLE_CATEGORIES.get(cls_id, "❓ INDEFINIDO")
            categories_count[category] = categories_count.get(category, 0) + 1
        
        # Mostra estatísticas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 Itens encontrados:**")
            for (_, _, _, _), cls_id, conf in detections:
                class_name = RECYCLABLE_CLASSES.get(cls_id, f"Classe {cls_id}")
                category = RECYCLE_CATEGORIES.get(cls_id, "❓ INDEFINIDO")
                st.write(f"• **{class_name}** — {conf:.1%}")
        
        with col2:
            st.markdown("**🗂️ Por categoria:**")
            for category, count in categories_count.items():
                st.write(f"• {category}: **{count}** {'item' if count == 1 else 'itens'}")
            
            if categories_count:
                st.success(f"♻️ **Total: {sum(categories_count.values())} materiais recicláveis detectados!**")

elif not model:
    st.error("❌ Falha ao carregar o modelo. Tente recarregar a página.")
