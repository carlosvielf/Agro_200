from ultralytics import YOLO

    
model = YOLO('yolo11n.pt')


results = model.train(data='/home/carlos/workspace/synapse/agromerica/Reconhecimento_IA/dataset/data.yaml',epochs=200,device=0, batch=8, project = '/home/carlos/workspace/synapse/agromerica/Reconhecimento_IA/Models',name='agromerica_train')