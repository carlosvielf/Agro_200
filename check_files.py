import os

# Caminhos das subpastas
path_images = os.path.join('/home/carlos/workspace/synapse/agromerica/Reconhecimento_IA/dataset/agro_junta/images/val')
path_labels = os.path.join('/home/carlos/workspace/synapse/agromerica/Reconhecimento_IA/dataset/agro_junta/labels/val')

# Obtem os nomes dos arquivos sem extensão
image_files = {os.path.splitext(f)[0] for f in os.listdir(path_images) if f.endswith('.jpg')}
label_files = {os.path.splitext(f)[0] for f in os.listdir(path_labels) if f.endswith('.txt')}
print('Arquivos de imagem:', sorted(image_files))
print('Arquivos de label:', sorted(label_files))

# Verifica correspondência
only_in_images = image_files - label_files
only_in_labels = label_files - image_files

# Resultado
if not only_in_images and not only_in_labels:
    print("Todos os arquivos possuem correspondência entre imagens e labels.")
else:
    if only_in_images:
        print("Arquivos .jpg sem correspondência .txt:")
        for f in sorted(only_in_images):
            print(f"{f}.jpg")
    
    if only_in_labels:
        print("Arquivos .txt sem correspondência .jpg:")
        for f in sorted(only_in_labels):
            print(f"{f}.txt")
