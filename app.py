# app_garbage_detector.py
# Detector de Resíduos com YOLOv5

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import requests
import tempfile
from pathlib import Path
import numpy as np

# Configurações
MODEL_URL = "https://huggingface.co/keremberke/yolov5s-garbage/resolve/main/best.pt"
IMG_SIZE = 640
DEFAULT_CONF = 0.25

# Cores específicas para cada material
MATERIAL_COLORS = {
    "plastic": "#FFEB3B",    # Amarelo
    "glass": "#4CAF50",      # Verde
    "metal": "#607D8B",      # Azul-cinza
    "paper": "#8BC34A",      # Verde claro
    "cardboard": "#FF9800",  # Laranja
    "biodegradable": "#795548" # Marrom
}

def _hex2rgb(hexcode: str) -> tuple[int, int, int]:
    hexcode = hexcode.lstrip("#")
    return tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))

@st.cache_resource(show_spinner="🔄 Carregando modelo...")
def load_model():
    """Carrega modelo de detecção."""
    try:
        from ultralytics import YOLO
        
        # Usar YOLOv8 nano - mais estável para Streamlit Cloud
        model = YOLO('yolov8n.pt')
        return model
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None

def detect(model, image, conf_threshold):
    """Executa detecção de materiais recicláveis na imagem."""
    if model is None:
        return []
    
    try:
        # Executa detecção
        results = model(image, conf=conf_threshold, verbose=False)
        
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
                    
                    # Mapear classes COCO para materiais
                    material_id = map_coco_to_material(cls)
                    if material_id is not None:
                        detections.append(((x1, y1, x2, y2), material_id, conf))
        
        return detections
        
    except Exception as e:
        st.error(f"❌ Erro na detecção: {e}")
        return []

def draw_boxes(img: Image.Image, detections, class_names):
    """Desenha bounding boxes na imagem com cores específicas por material."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), cls_id, conf in detections:
        # Usar cor específica do material
        color_hex = MATERIAL_COLOR_MAP.get(cls_id, "#FF595E")
        color = _hex2rgb(color_hex)
        label = f"{class_names[cls_id]} {conf:.0%}"

        # Caixa
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Texto
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        draw.text((x1, y1 - th), label, fill="white", font=font)

    return img

def map_coco_to_material(coco_class):
    """Mapeia classes COCO para categorias de materiais."""
    # Mapeamento de objetos COCO para materiais mais prováveis
    coco_to_material = {
        # Garrafas e recipientes -> geralmente plástico
        39: 5,  # bottle -> plástico
        41: 5,  # cup -> plástico
        45: 5,  # bowl -> plástico
        
        # Utensílios metálicos
        42: 3,  # fork -> metal
        43: 3,  # knife -> metal  
        44: 3,  # spoon -> metal
        
        # Papel
        73: 4,  # book -> papel
        
        # Vidro
        40: 2,  # wine glass -> vidro
        
        # Outros objetos metálicos/eletrônicos
        74: 3,  # clock -> metal
        76: 3,  # scissors -> metal
        63: 3,  # laptop -> metal
        64: 3,  # mouse -> metal
        67: 3,  # cell phone -> metal
    }
    
    return coco_to_material.get(coco_class)

# ───────────────────────────── UI ────────────────────────────────
st.set_page_config(page_title="Detector de Recicláveis", layout="wide")
st.title("♻️ Detector de Materiais Recicláveis")
st.markdown("**Detecta o MATERIAL dos resíduos: plástico, vidro, metal, papel, papelão**")

# Controle de confiança
conf_slider = st.sidebar.slider(
    "Confiança mínima (%)", 5, 90, int(DEFAULT_CONF * 100), 1
) / 100

# Carrega modelo
model = load_model()

# Classes de materiais recicláveis (do modelo garbage)
MATERIAL_CLASSES = {
    0: "🌱 Biodegradável",
    1: "📦 Papelão", 
    2: "� Vidro",
    3: "🥫 Metal",
    4: "� Papel",
    5: "🥤 Plástico"
}

# Cores por material
MATERIAL_COLOR_MAP = {
    0: "#795548",  # Marrom - Biodegradável
    1: "#FF9800",  # Laranja - Papelão
    2: "#4CAF50",  # Verde - Vidro
    3: "#607D8B",  # Azul-cinza - Metal
    4: "#8BC34A",  # Verde claro - Papel
    5: "#FFEB3B"   # Amarelo - Plástico
}

# Instruções de reciclagem
RECYCLE_INSTRUCTIONS = {
    0: "� ORGÂNICO - Compostagem",
    1: "� PAPELÃO - Lixeira de papel",
    2: "� VIDRO - Lixeira de vidro", 
    3: "� METAL - Lixeira de metal",
    4: "🟢 PAPEL - Lixeira de papel",
    5: "🟡 PLÁSTICO - Lixeira de plástico"
}

if model:
    st.sidebar.markdown("**🔍 Materiais detectados:**")
    st.sidebar.markdown("🟡 **🥤 Plástico** - Garrafas PET, sacolas, embalagens")
    st.sidebar.markdown("🟢 **🍾 Vidro** - Garrafas, potes, vidros")  
    st.sidebar.markdown("🔵 **🥫 Metal** - Latas, alumínio, ferro")
    st.sidebar.markdown("🟢 **📄 Papel** - Jornais, revistas, folhas")
    st.sidebar.markdown("� **📦 Papelão** - Caixas, embalagens")
    st.sidebar.markdown("� **🌱 Biodegradável** - Orgânicos, compostáveis")
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
                "- **Materiais detectados:** plástico, vidro, metal, papel, papelão, biodegradável\n"
                "- Aproxime-se dos objetos para melhor detecção\n"
                "- Evite fundos muito complexos"
            )
    else:
        img_bb = draw_boxes(img.copy(), detections, MATERIAL_CLASSES)
        st.image(img_bb, caption="Detecções", use_container_width=True)

        st.subheader("📊 Detalhes das Detecções")
        
        # Contador por categoria
        categories_count = {}
        for (_, _, _, _), cls_id, conf in detections:
            category = RECYCLE_INSTRUCTIONS.get(cls_id, "❓ INDEFINIDO")
            categories_count[category] = categories_count.get(category, 0) + 1
        
        # Mostra estatísticas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 Itens encontrados:**")
            for (_, _, _, _), cls_id, conf in detections:
                class_name = MATERIAL_CLASSES.get(cls_id, f"Classe {cls_id}")
                category = RECYCLE_INSTRUCTIONS.get(cls_id, "❓ INDEFINIDO")
                st.write(f"• **{class_name}** — {conf:.1%}")
        
        with col2:
            st.markdown("**🗂️ Por categoria:**")
            for category, count in categories_count.items():
                st.write(f"• {category}: **{count}** {'item' if count == 1 else 'itens'}")
            
            if categories_count:
                st.success(f"♻️ **Total: {sum(categories_count.values())} materiais recicláveis detectados!**")

elif not model:
    st.error("❌ Falha ao carregar o modelo. Tente recarregar a página.")
