# app_trashnet_bbox.py
# pip install streamlit transformers pillow torch

import streamlit as st
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch

# ───────────────────────── configuração ─────────────────────────
MODEL_NAME = "google/owlvit-base-patch32"
CLASSES = ["glass bottle", "plastic bottle", "metal can",
           "paper", "cardboard", "trash bag"]
SCORE_THRESHOLD = 0.35            # confiança mínima p/ exibir caixa

# ─────────────────────── carga do modelo ────────────────────────
@st.cache_resource(show_spinner="🔄 Baixando modelo…")
def load_model():
    mdl = OwlViTForObjectDetection.from_pretrained(MODEL_NAME)
    proc = OwlViTProcessor.from_pretrained(MODEL_NAME)
    mdl.eval()
    return mdl, proc

model, processor = load_model()

# ───────────────────────── funções util ─────────────────────────
@torch.no_grad()
def detect(image: Image.Image):
    """Roda o OwlViT e devolve caixas, rótulos e scores acima do threshold."""
    inputs = processor(text=[CLASSES], images=image, return_tensors="pt")
    outputs = model(**inputs)
    # pós-processa p/ coordenadas no sistema xyxy pixel
    target_sizes = torch.tensor([image.size[::-1]])  # (h, w)
    results = processor.post_process_object_detection(
        outputs, threshold=SCORE_THRESHOLD, target_sizes=target_sizes
    )[0]  # lista de dicts

    bboxes, labels, scores = results["boxes"], results["labels"], results["scores"]
    return list(zip(bboxes, labels, scores))

def draw_boxes(image: Image.Image, detections):
    """Desenha bounding boxes na imagem."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:  # fallback caso a fonte não exista
        font = ImageFont.load_default()

    for box, lab_idx, score in detections:
        x0, y0, x1, y1 = map(int, box.tolist())
        cls_name = CLASSES[lab_idx]
        caption = f"{cls_name} {score:.0%}"
        # caixa
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        text_w, text_h = draw.textsize(caption, font)
        draw.rectangle([x0, y0 - text_h, x0 + text_w, y0], fill="red")
        draw.text((x0, y0 - text_h), caption, fill="white", font=font)
    return image

# ─────────────────────── interface Streamlit ────────────────────
st.title("♻️ Classificador de Resíduos com Bounding Box")
st.markdown(
    "Capture uma imagem; o modelo identificará **vidro, plástico, metal, papel, "
    "papelão ou saco de lixo** e mostrará as caixas na foto."
)

cam_img = st.camera_input("Tire a foto e aguarde a predição")

if cam_img:
    img = Image.open(cam_img).convert("RGB")        # garante modo RGB
    st.image(img, caption="Imagem capturada", width=600)

    with st.spinner("🔎 Detectando resíduos…"):
        dets = detect(img)

    if not dets:
        st.warning("Nenhum objeto correspondente encontrado 😕")
    else:
        img_boxes = draw_boxes(img.copy(), dets)
        st.image(img_boxes, caption="Resultado com Bounding Boxes", width=600)

        # lista detalhada
        for _, lab_idx, score in dets:
            st.write(
                f"• **{CLASSES[lab_idx]}** — confiança {score:.1%}"
            )
