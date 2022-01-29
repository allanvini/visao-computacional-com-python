import cv2
import time
import threading
import requests

# Carrega o conteudo para análise (pode ser uma câmera ou um vídeo em um arquivo)
cap = cv2.VideoCapture('test_data/camera.mp4');

#Identifica a resolução do vídeo (será usado para definir a resolução da saída)
vwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
vheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (vwidth, vheight)

# Inicializa a saída do vídeo
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, size)

# Seleciona o modelo que será usado na detecção
net = cv2.dnn.readNet('rede_pre_treinada/yolov4.weights', 'rede_pre_treinada/yolov4.cfg')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

# Processa o video quadro a quadro fazendo a detecção dos veículos

while (cap.isOpened()):
    _, frame = cap.read()

    #Realiza o scan no quadro
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    #contador global de veículos
    vehicleCounter = 0;

    for(classid, score, box) in zip(classes, scores, boxes):

        # Filtra as classes que irão ser consideradas (2 para carros, 3 para motocicletas, 6 para onibus e 7 para caminhoes)
        # E também a precisão minima de 60%
        if (classid == 2 or classid == 3 or classid == 6 or classid == 7 and score > .60):
            vehicleCounter+=1
            color = (0,255,150)
            cv2.rectangle(frame, box, color, 1)

            #Escreve acima da box a precisão da detecção
            cv2.putText(frame, f"{str(round((score*100), 2))}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 2)
        
    # Imprime no frame a contagem de veículos
    vehicle_label = f"Veiculos: {vehicleCounter}"
    cv2.putText(frame, vehicle_label, (500,125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    #Grava o quadro no vídeo de saída
    out.write(frame)

    cv2.imshow('detections', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
