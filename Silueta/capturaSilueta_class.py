import cv2 as cv
import numpy as np
import os
import imutils
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class CapturaSilueta(QThread):
    def __init__(self, name, filepath, label_video):
        super().__init__()
        self.personName = name
        self.filepath=filepath
        self.label_video=label_video
        dataPath = 'Data_Silueta' 
        self.personPath = dataPath + '/' + self.personName
        if not os.path.exists(self.personPath):
            print('Carpeta creada: ',self.personPath)
            os.makedirs(self.personPath)
        # Video que se ubica en el fondo
        prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
        # Weights
        model = "model/MobileNetSSD_deploy.caffemodel"
        self.classes = {0:"background", 1:"aeroplane", 2:"bicycle",
                3:"bird", 4:"boat",
                5:"bottle", 6:"bus",
                7:"car", 8:"cat",
                9:"chair", 10:"cow",
                11:"diningtable", 12:"dog",
                13:"horse", 14:"motorbike",
                15:"person", 16:"pottedplant",
                17:"sheep", 18:"sofa",
                19:"train", 20:"tvmonitor"}
        self.net = cv.dnn.readNetFromCaffe(prototxt, model)
        self.cap = cv.VideoCapture(self.filepath)
    Image_salida_upd=pyqtSignal(QImage)
    def run(self):
        self.hilo_corriendo=True
        count = 0
        BG_COLOR = (255, 255, 255)
        while True:
            ret, frame, = self.cap.read()
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
            self.net.setInput(blob)
            detections = self.net.forward()
            auxFrame = thresh.copy()
            for detection in detections[0][0]:
                if self.classes[detection[1]]!="person":
                        continue
                if detection[2] > 0.45:
                    label = self.classes[detection[1]]
                    box = detection[3:7] * [width, height, width, height]
                    x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    body=auxFrame[y_start:y_end,x_start:x_end]
                    #print("cuerpo",body)
                    
                    cv.rectangle(thresh, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    cv.imwrite(self.personPath +'/body_{}.png'.format(count),body)
                    count = count + 1
            frame =  imutils.resize(thresh, width=700)
            Image=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            convertir_QT=QImage(Image.data,Image.shape[1],Image.shape[0],QImage.Format_RGB888)
            pic=convertir_QT.scaled(self.label_video.width(),self.label_video.height(),Qt.KeepAspectRatio)
            self.Image_salida_upd.emit(pic)
            if self.hilo_corriendo==False:
                break
        self.cap.release()
        cv.destroyAllWindows()
    def stop(self):
        self.hilo_corriendo=False
        self.quit()