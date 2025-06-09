from ultralytics import YOLO

def main():
    # Carrega o modelo base YOLOv8 nano
    model = YOLO('yolov8n.pt')

    print("Iniciando o treinamento do modelo...")
    results = model.train(
        data='data.yaml',
        epochs=50,
        device=0,
        cache=True  # Ativa o cache de imagens/anotações para acelerar o treinamento
    )

    print("Treinamento concluído com sucesso!")
    print(f"O melhor modelo foi salvo no diretório: {results.save_dir}")
    print("Você pode encontrar o modelo em 'runs/detect/train/weights/best.pt'")

if __name__ == '__main__':
    main()
