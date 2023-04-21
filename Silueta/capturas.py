import cv2 as cv
import numpy as np
import os
import imutils
personName = 'Angel'
dataPath = 'Data_Silueta' 
personPath = dataPath + '/' + personName
if not os.path.exists(personPath):
	print('Carpeta creada: ',personPath)
	os.makedirs(personPath)
# Video que se ubica en el fondo
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
# Weights
model = "model/MobileNetSSD_deploy.caffemodel"
classes = {0:"background", 1:"aeroplane", 2:"bicycle",
          3:"bird", 4:"boat",
          5:"bottle", 6:"bus",
          7:"car", 8:"cat",
          9:"chair", 10:"cow",
          11:"diningtable", 12:"dog",
          13:"horse", 14:"motorbike",
          15:"person", 16:"pottedplant",
          17:"sheep", 18:"sofa",
          19:"train", 20:"tvmonitor"}
net = cv.dnn.readNetFromCaffe(prototxt, model)
cap = cv.VideoCapture('pruebas/Angel1_Cam1.mp4')
count = 0
BG_COLOR = (255, 255, 255)
while True:
    ret, frame, = cap.read()
    if ret == False:
          break
    height, width, _ = frame.shape
  
    frame_resized = cv.resize(frame, (300, 300))
    blob = cv.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray=cv.medianBlur(gray,9)
    _,thresh = cv.threshold(gray, 155, 255, cv.THRESH_BINARY_INV)
    thresh  = thresh.astype(np.uint8)
    thresh =cv.medianBlur(thresh ,9)
    
   
    #thresh = thresh.astype(np.uint8)
    #thresh  = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv.THRESH_BINARY_INV,99,3)
    #gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    #thresh = cv.cvtColor(thresh[1], cv.COLOR_GRAY2BGR)
    net.setInput(blob)
    detections = net.forward()
    auxFrame = thresh.copy()
    for detection in detections[0][0]:
          if classes[detection[1]]!="person":
                continue
          if detection[2] > 0.45:
               label = classes[detection[1]]
               box = detection[3:7] * [width, height, width, height]
               x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
               
               body=auxFrame[y_start:y_end,x_start:x_end]
               #print("cuerpo",body)
            
               cv.rectangle(thresh, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
               cv.imshow("salida",body)
               #cv.imwrite(personPath +'/body_{}.png'.format(count),body)
               count = count + 1
    frame =  imutils.resize(thresh, width=700)
    cv.imshow("Frame", frame)
    k=cv.waitKey(1)
    if k==27 or count>=300:
        break
cap.release()
cv.destroyAllWindows()