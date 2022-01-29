import cv2
import time
import threading
import requests

# inicializa o conteudo para análise da biblioteca

class_names = []
with open('coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture('gta.mp4');

net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)

vehicleCounter = 0

###################################################################################################
# Inicia uma thread de loop que vai atualizando a API com os dados

def updater():
    myTimer = 'true'
    while myTimer == 'true':
        requests.post('http://localhost:4001/data/register', json={'cars': vehicleCounter})
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

###################################################################################################



# Processa o video quadro a quadro fazendo a detecção dos veículos

while True:
    
    _, frame = cap.read()
    start = time.time()

    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    end = time.time()

    vehicleCounter = 0;
    for(classid, score, box) in zip(classes, scores, boxes):
        if classid == 2:
            vehicleCounter+=1
            color = (0,255,150)
            cv2.rectangle(frame, box, color, 1)
        
        

    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
    vehicle_label = f"Veiculos: {vehicleCounter}"

    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, vehicle_label, (500,125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow('detections', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
