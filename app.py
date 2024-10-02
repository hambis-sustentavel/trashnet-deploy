import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Nome do modelo no Hugging Face
model_name = "edwinpalegre/ee8225-group4-vit-trashnet-enhanced"

# Carregar o modelo e o feature extractor
model = ViTForImageClassification.from_pretrained(model_name)
image_processor = ViTImageProcessor.from_pretrained(model_name)

# Colocar o modelo em modo de avaliação
model.eval()

# Função para fazer a inferência na imagem
def predict_image_class(image):
    # Processar a imagem usando o ViTImageProcessor
    inputs = image_processor(images=image, return_tensors="pt")
    
    # Fazer a inferência
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obter as previsões
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Retornar a classe prevista
    return model.config.id2label[predicted_class_idx]

# Interface do usuário com Streamlit
st.title("Classificador de Materiais - TrashNet")

# Capturar a imagem da câmera
camera_image = st.camera_input("Capture uma imagem usando sua câmera")

if camera_image is not None:
    # Abrir a imagem capturada
    image = Image.open(camera_image)
    
    # Exibir a imagem na interface do Streamlit
    st.image(image, caption='Imagem capturada.', use_column_width=True)
    
    # Rodar a classificação
    st.write("Classificando...")
    predicted_class = predict_image_class(image)
    
    # Exibir o resultado da classificação
    st.write(f"Classe prevista: {predicted_class}")