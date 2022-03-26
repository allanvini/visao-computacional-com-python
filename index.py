import cv2
import threading
import requests
import time

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

# Seta o modelo de detecção no OpenCV
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

#Inicializa a variavel global de contagem
vehicleCounter = 0;


# Definimos a URL da nossa API que irá receber os dados
api = 'http://localhost:4001/data/register'

# Inicializa o looping de posts para a API
def updater():
    myTimer = 'true'
    while myTimer == 'true':
        requests.post(api, json={'cars': vehicleCounter})
        time.sleep(10)

updater_threading = threading.Thread(target=updater)
updater_threading.start()

def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    timer = threading.Timer(sec, func_wrapper)
    timer.start()
    return timer


# Loop para percorrer os quadros
while (cap.isOpened()):

    #Captura o quadro
    _, frame = cap.read()

    #Realiza o scan no quadro
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    #variavel contadora de detecções
    counter = 0

    #Percorre as detecções feitas no quadro
    for(classid, score, box) in zip(classes, scores, boxes):
        # Filtra as classes que irão ser consideradas (2 para carros, 3 para motocicletas, 6 para onibus e 7 para caminhoes)
        # E também a precisão minima de 50%
        if (classid == 2 or classid == 3 or classid == 6 or classid == 7 and score > .50):
            counter+=1
            color = (0,255,150)
            cv2.rectangle(frame, box, color, 1)

            #Escreve acima da box a precisão da detecção
            cv2.rectangle(frame, (box[0]-10, box[1]-25, 75, 25), (0,0,0), -1)

            cv2.putText(frame, f"{str(round((score*100), 1))}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
    
    #seta a contagem na variavel global
    vehicleCounter = counter

    # Imprime no frame a contagem de veículos
    vehicle_label = f"Veiculos: {vehicleCounter}"
    cv2.rectangle(frame, (0,0), (100,25), (0,0,0), -1)
    cv2.putText(frame, vehicle_label, (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)

    #Grava o quadro no vídeo de saída
    out.write(frame)

    #Mostra o quadro detectado
    cv2.imshow('detections', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
