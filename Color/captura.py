import cv2
import imutils
import os
personName = 'Flor'
dataPath = 'Data_Color' 
personPath = dataPath + '/' + personName
if not os.path.exists(personPath):
	print('Carpeta creada: ',personPath)
	os.makedirs(personPath)
# Video que se ubica en el fondo
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
# Weights
model = "model/MobileNetSSD_deploy.caffemodel"

# Class labels
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

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ----------- READ THE IMAGE AND PREPROCESSING -----------
cap = cv2.VideoCapture("pruebas/Flor2_cam1.mp4")
#cap = cv2.VideoCapture ('http://10.252.190.80:8080/video')
count = 0
while True:
     ret, frame, = cap.read()
     if ret == False:
          break
     #frame = cv2.flip(frame, 1)
     height, width, _ = frame.shape
    
     frame_resized = cv2.resize(frame, (300, 300))

     # Create a blob
     blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
     #print("blob.shape:", blob.shape)
     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     im3 = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     # ----------- DETECTIONS AND PREDICTIONS -----------
     net.setInput(blob)
     detections = net.forward()
     auxFrame = im3.copy()
     #print(detections)
     for detection in detections[0][0]:
          if classes[detection[1]]!="person":
                continue
          
          if detection[2] > 0.45:
               label = classes[detection[1]]
               box = detection[3:7] * [width, height, width, height]
               x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
               
               body=auxFrame[y_start:y_end,x_start:x_end]
               #print("cuerpo",body)
            
               
               cv2.rectangle(im3, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
              
               cv2.imwrite(personPath +'/body_{}.png'.format(count),body)
               count = count + 1
     frame =  imutils.resize(im3, width=700)
     
     k=cv2.waitKey(1)
     if k==27 or count>=100:
          break
cap.release()
cv2.destroyAllWindows()