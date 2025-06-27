# app_garbage_detector.py
# Requisitos:
#   pip install streamlit yolov5 pillow torch

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import yolov5                          # biblioteca recomendada pelo autor do modelo :contentReference[oaicite:0]{index=0}
import torch

# ───────────────────────── Configurações ─────────────────────────
MODEL_REPO = "keremberke/yolov5s-garbage"
IMG_SIZE   = 640           # resolução em que o YOLOv5 fará a inferência
CONF_THRES = 0.25          # confiança mínima inicial (pode ser alterada na UI)
IOU_THRES  = 0.45          # limiar IoU para NMS

# ──────────────────────── Carregamento do modelo ─────────────────
@st.cache_resource(show_spinner="🔄 Baixando e carregando modelo…")
def load_model():
    model = yolov5.load(MODEL_REPO)     # puxa o checkpoint direto do HF
    model.conf = CONF_THRES
    model.iou  = IOU_THRES
    return model

model   = load_model()
CLASSES = model.names                   # ['biodegradable', 'cardboard', 'glass', …] :contentReference[oaicite:1]{index=1}

# ─────────────────────── Funções auxiliares ─────────────────────
@torch.no_grad()
def detect(image):
    """Roda inferência YOLOv5 e devolve caixas (xyxy), rótulos e scores."""
    results = model(image, size=IMG_SIZE)
    preds   = results.pred[0]
    
    # Debug: mostrar todas as detecções antes do filtro
    if preds is not None and preds.size(0) > 0:
        all_scores = preds[:, 4].cpu().numpy()
        st.sidebar.write(f"Total detecções brutas: {len(all_scores)}")
        st.sidebar.write(f"Scores máx/mín: {all_scores.max():.3f}/{all_scores.min():.3f}")
        
        # Filtrar por confiança
        mask = all_scores >= model.conf
        filtered_preds = preds[mask]
        st.sidebar.write(f"Após filtro conf={model.conf:.2f}: {filtered_preds.size(0)}")
        
        if filtered_preds.size(0) == 0:
            return []
            
        boxes  = filtered_preds[:, :4].cpu().numpy()
        scores = filtered_preds[:, 4].cpu().numpy()
        labels = filtered_preds[:, 5].cpu().numpy().astype(int)
        return list(zip(boxes, labels, scores))
    else:
        st.sidebar.write("Nenhuma detecção bruta encontrada")
        return []

def draw_boxes(img, detections):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), lab_idx, score in detections:
        caption = f"{CLASSES[lab_idx]} {score:.0%}"
        # caixa
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # fundo do rótulo
        bbox = draw.textbbox((0, 0), caption, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill="red")
        draw.text((x1, y1 - th), caption, fill="white", font=font)
    return img

# ─────────────────────── Interface Streamlit ────────────────────
st.title("🗑️ Detector de Resíduos (YOLOv5s-Garbage)")
st.caption(
    "Modelo de detecção treinado para resíduos sólidos — identiﬁca vidro, plástico, papel, "
    "papelão, metal e biodegradáveis."
)

# ⬆️ Ajuste de confiança opcional
conf = st.slider("Confiança mínima (%)", 5, 90, int(CONF_THRES * 100), 1) / 100
model.conf = conf

# 📊 Debug info
st.sidebar.subheader("🔧 Debug Info")
st.sidebar.write(f"Classes disponíveis: {len(CLASSES)}")
st.sidebar.write("Classes:", CLASSES)

# 📸 Captura da câmera
st.subheader("📸 Captura de Imagem")
img_source = st.radio("Escolha a fonte:", ["📷 Câmera", "🖼️ Upload de arquivo"])

if img_source == "📷 Câmera":
    img_file = st.camera_input("Tire uma foto do resíduo e aguarde…")
else:
    img_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Imagem capturada", use_container_width=True)
    
    # Mostrar info da imagem
    st.sidebar.write(f"Tamanho da imagem: {img.size}")

    with st.spinner("Detectando…"):
        detections = detect(img)

    if not detections:
        st.warning("Nenhum resíduo detectado acima do limiar de confiança.")
        st.info("💡 **Dicas para melhorar a detecção:**")
        st.write("• Diminua o slider de confiança para 5-15%")
        st.write("• Certifique-se que há objetos de lixo visíveis na foto")
        st.write("• Tente com boa iluminação")
        st.write("• Aproxime-se mais dos objetos")
        st.write("• Tipos suportados: vidro, plástico, papel, papelão, metal, biodegradáveis")
    else:
        img_bbox = draw_boxes(img.copy(), detections)
        st.image(img_bbox, caption="Detecções", use_container_width=True)

        # resultados tabulados
        st.subheader("Detalhe das detecções")
        for (_, _, _, _), lab_idx, score in detections:
            st.write(f"- **{CLASSES[lab_idx]}** – confiança {score:.1%}")
