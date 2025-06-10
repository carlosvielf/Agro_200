import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import os

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Agromérica",
    page_icon="🌱",
    layout="wide"
)

# --- GERENCIAMENTO DE ESTADO ---
# Usamos o st.session_state para "lembrar" se a detecção já foi executada.
if 'recognition_done' not in st.session_state:
    st.session_state.recognition_done = False

# Função para resetar o estado se uma nova imagem for enviada
def reset_state():
    st.session_state.recognition_done = False

# --- FUNÇÃO DE CACHE PARA O MODELO ---
@st.cache_resource
def load_model(model_path):
    """Carrega o modelo YOLO a partir do caminho especificado."""
    if not os.path.exists(model_path):
        st.error(f"Caminho do modelo não encontrado: {model_path}")
        st.stop()
    model = YOLO(model_path)
    return model

# --- INTERFACE PRINCIPAL ---
st.title(" Reconhecimento de peças - Agromérica")
st.markdown("Faça o upload de uma imagem e o modelo treinado fará a detecção dos objetos.")

model_path = '/home/fifo/Área de trabalho/projeto_agromerica/Reconhecimento_IA/Models/agromerica_train3/weights/best.pt'

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Widget para upload de imagem, que reseta o estado ao trocar de arquivo
uploaded_file = st.file_uploader(
    "Escolha uma imagem...",
    type=["jpg", "jpeg", "png"],
    on_change=reset_state
)

if uploaded_file is not None:
    image_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_data))

    # Se a detecção ainda NÃO foi feita, exibe a imagem original e o botão
    if not st.session_state.recognition_done:
        # A imagem original é exibida com uma largura fixa de 600 pixels
        st.image(image, caption="Imagem Original", width=600)
        
        if st.button("Executar Reconhecimento"):
            # Apenas muda o estado e força o recarregamento da página
            st.session_state.recognition_done = True
            st.rerun()

    # Se a detecção JÁ foi feita, exibe o resultado
    else:
        with st.spinner("Processando..."):
            # Realiza a predição
            results = model(image)
            # Prepara a imagem com as anotações
            annotated_image_rgb = Image.fromarray(results[0].plot()[..., ::-1])

            st.subheader("Imagem com Detecção")
            # A imagem com detecção também é exibida com uma largura fixa de 600 pixels
            st.image(annotated_image_rgb, caption="Imagem Processada", width=600)

            st.subheader("Resultados")

            # Exibe os resultados da detecção
            if results and results[0].boxes:
                names = results[0].names
                for box in results[0].boxes:
                    class_name = names.get(int(box.cls), f"Classe {int(box.cls)}")
                    confidence = box.conf.item() * 100
                    st.write(f"- **Classe:** {class_name}, **Confiança:** {confidence:.2f}%")
            else:
                st.write("Nenhum objeto detectado na imagem.")

            st.divider()

            # Exibe os detalhes estáticos do item
            st.markdown(f"**Código:** 02RV-0512")
            st.markdown(f"**Descrição:** JUNTA CRIA 115FL X 302PS")
            st.markdown(f"**Modelo da Máquina:** Arrancadora HP")