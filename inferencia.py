# from ultralytics import YOLO

# # Load a model
# model = YOLO("/home/fifo/Área de trabalho/synapse/runs/detect/train9/weights/best.pt")  # load a custom model

# # Predict with the model
# results = model("/home/fifo/Área de trabalho/projeto_agromerica/dataset/frame_00036.jpg")  # predict on an image

# # Access the results
# for result in results:
#     xywh = result.boxes.xywh  # center-x, center-y, width, height
#     xywhn = result.boxes.xywhn  # normalized
#     xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
#     xyxyn = result.boxes.xyxyn  # normalized
#     names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
#     confs = result.boxes.conf  # confidence score of each box

import cv2
from ultralytics import YOLO

# Carrega o modelo
model = YOLO("/home/fifo/Área de trabalho/synapse/runs/detect/train9/weights/best.pt")

# Caminho da imagem
image_path = "/home/fifo/Área de trabalho/projeto_agromerica/dataset/agro_junta/images/train/0c048022-frame_00540.jpg"

# Realiza a predição
results = model(image_path)

for result in results:
    # Obtém imagem com boxes
    img_with_boxes = result.plot()

    # Mostra a imagem
    cv2.imshow("Detecção", img_with_boxes)
    cv2.imwrite("detecao_salva.jpg", img_with_boxes)  # opcional: salva o resultado
    cv2.waitKey(0)
    cv2.destroyAllWindows()
