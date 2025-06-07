import cv2
from ultralytics import YOLO
import os

def main():
    """
    Script para fazer predições em uma nova imagem usando um modelo YOLOv8 treinado.
    """
    # --- CONFIGURAÇÕES ---
    # Caminho para o modelo treinado (o arquivo 'best.pt').
    # Altere este caminho para o local onde seu treinamento salvou o modelo.
    model_path = 'runs/detect/train/weights/best.pt'

    # Caminho para a imagem ou pasta de imagens que você quer analisar.
    image_to_predict = 'caminho/para/sua/imagem_de_teste.jpg' # <-- MUDE AQUI

    # --- VERIFICAÇÕES ---
    if not os.path.exists(model_path):
        print(f"Erro: Arquivo do modelo não encontrado em '{model_path}'")
        print("Verifique se o treinamento foi concluído e o caminho está correto.")
        return

    if not os.path.exists(image_to_predict):
        print(f"Erro: Imagem de teste não encontrada em '{image_to_predict}'")
        return

    # --- EXECUÇÃO ---
    print("Carregando modelo treinado...")
    model = YOLO(model_path)

    print(f"Executando detecção na imagem: {image_to_predict}")
    # Executa a predição na imagem
    results = model(image_to_predict)

    # Plota os resultados (desenha as caixas delimitadoras na imagem)
    annotated_frame = results[0].plot()

    # Define o nome do arquivo de saída
    output_filename = "resultado_deteccao.jpg"

    # Salva a imagem com as detecções
    cv2.imwrite(output_filename, annotated_frame)
    print(f"Imagem com as detecções foi salva como '{output_filename}'")

    # Mostra a imagem em uma janela
    cv2.imshow("Deteccao de Pecas - Pressione qualquer tecla para sair", annotated_frame)
    
    # Espera o usuário pressionar uma tecla para fechar a janela da imagem
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()