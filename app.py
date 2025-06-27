# app_garbage_detector.py
# Requisitos:
#   pip install streamlit torch pillow

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from pathlib import Path
import tempfile
import requests

# ───────────────────────── Configurações ─────────────────────────
WEIGHTS_URL = (
    "https://huggingface.co/keremberke/yolov5s-garbage"
    "/resolve/main/weights/best.pt"
)
YOLO_REPO   = "ultralytics/yolov5"   # repo oficial no GitHub
IMG_SIZE    = 640
DEFAULT_CONF, DEFAULT_IOU = 0.25, 0.45

# Paleta simples por classe (hex → RGB)
COLORS = [
    "#FF595E", "#FFCA3A", "#8AC926",
    "#1982C4", "#6A4C93", "#FF924C"
]


# ────────────────────── utilitários auxiliares ───────────────────
def _hex2rgb(hexcode: str) -> tuple[int, int, int]:
    hexcode = hexcode.lstrip("#")
    return tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))


@st.cache_resource(show_spinner="🔄 Baixando/Carregando modelo…")
def load_model(conf=DEFAULT_CONF, iou=DEFAULT_IOU):
    """
    Baixa o repositório Ultralytics, carrega os pesos .pt hospedados no Hugging Face
    e devolve o modelo pronto para inferência.
    """
    try:
        model = torch.hub.load(
            YOLO_REPO,
            "custom",
            path=WEIGHTS_URL,
            trust_repo=True,
        )
    except Exception as e:
        st.error(
            "❌ Falha ao baixar o modelo. "
            "Verifique conexão/URL ou tente novamente mais tarde."
        )
        raise e

    model.conf = conf
    model.iou = iou
    return model


def detect(model, image):
    """Executa inferência YOLOv5 e devolve detecções filtradas (xyxy, label, score)."""
    with torch.no_grad():
        results = model(image, size=IMG_SIZE)
    preds = results.pred[0]

    if preds is None or preds.size(0) == 0:
        return []

    boxes = preds[:, :4].cpu().numpy()
    scores = preds[:, 4].cpu().numpy()
    labels = preds[:, 5].cpu().numpy().astype(int)
    return list(zip(boxes, labels, scores))


def draw_boxes(img: Image.Image, dets, class_names):
    """Desenha bounding-boxes + rótulos na imagem."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), lab, score in dets:
        color = _hex2rgb(COLORS[lab % len(COLORS)])
        label = f"{class_names[lab]} {score:.0%}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Usar textbbox em vez de textsize (depreciado)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        draw.text((x1, y1 - th), label, fill="white", font=font)

    return img


# ───────────────────────────── UI ────────────────────────────────
st.set_page_config(page_title="Detector de Resíduos", layout="wide")
st.title("🗑️ Detector de Resíduos (YOLOv5s-Garbage)")

# Controle de confiança
conf_slider = st.sidebar.slider(
    "Confiança mínima (%)", 5, 90, int(DEFAULT_CONF * 100), 1
) / 100

# Carrega modelo (em cache)
model = load_model(conf_slider, DEFAULT_IOU)
CLASSES = model.names

st.sidebar.markdown(f"**Classes:** {', '.join(CLASSES.values())}")

# Fonte da imagem
source = st.radio("Escolha a fonte da imagem:", ["📷 Câmera", "🖼️ Upload"])
file = (
    st.camera_input("Tire uma foto…")
    if source == "📷 Câmera"
    else st.file_uploader("Envie uma imagem", type=["png", "jpg", "jpeg"])
)

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Imagem original", use_container_width=True)

    with st.spinner("Detectando…"):
        detections = detect(model, img)

    if not detections:
        st.warning("Nenhum resíduo detectado acima do limiar escolhido.")
        with st.expander("💡 Dicas para melhores resultados"):
            st.write(
                "- Diminua o slider de confiança\n"
                "- Certifique-se de que o objeto aparece inteiro na imagem\n"
                "- Use boa iluminação"
            )
    else:
        img_bb = draw_boxes(img.copy(), detections, CLASSES)
        st.image(img_bb, caption="Detecções", use_container_width=True)

        st.subheader("Detalhes")
        for (_, _, _, _), lab, score in detections:
            st.write(f"- **{CLASSES[lab]}** — {score:.1%}")
