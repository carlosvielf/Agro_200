from ultralytics import YOLO

def main():
 
    model = YOLO('yolov8n.pt')

    print("Iniciando o treinamento do modelo...")
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        device='cpu'
    )

    print("Treinamento concluído com sucesso!")
    print(f"O melhor modelo foi salvo no diretório: {results.save_dir}")
    print("Você pode encontrar o modelo em 'runs/detect/train/weights/best.pt'")

if __name__ == '__main__':
    main()