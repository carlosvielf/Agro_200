import cv2
from ultralytics import YOLO
import os

def main():
    """
    Script para fazer predições em uma nova imagem usando um modelo YOLOv8 treinado.
    """
    # Caminho para o modelo treinado
    model_path = 'runs/detect/train/weights/best.pt'

    # Caminho da imagem para predição (ajuste conforme necessário)
    image_to_predict = 'dataset/frame_00036.jpg'

    if not os.path.exists(model_path):
        print(f"Erro: Arquivo do modelo não encontrado em '{model_path}'")
        return

    if not os.path.exists(image_to_predict):
        print(f"Erro: Imagem de teste não encontrada em '{image_to_predict}'")
        return

    print("Carregando modelo treinado...")
    model = YOLO(model_path)

    print(f"Executando detecção na imagem: {image_to_predict}")
    results = model(image_to_predict)

    annotated_frame = results[0].plot()
    output_filename = "resultado_deteccao.jpg"

    cv2.imwrite(output_filename, annotated_frame)
    print(f"Imagem com as detecções foi salva como '{output_filename}'")

    cv2.imshow("Deteccao de Pecas - Pressione qualquer tecla para sair", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
