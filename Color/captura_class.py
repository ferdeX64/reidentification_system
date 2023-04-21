from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import imutils
import os
class CapturaColor(QThread):
    def __init__(self, name, filepath, label_video):
        super().__init__()
        self.personName = name
        self.filepath=filepath
        self.label_video=label_video
        dataPath = 'Data_Color' 
        self.personPath = dataPath + '/' + self.personName
        if not os.path.exists(self.personPath):
            print('Carpeta creada: ',self.personPath)
            os.makedirs(self.personPath)
        # Video que se ubica en el fondo
        prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
        # Weights
        model = "model/MobileNetSSD_deploy.caffemodel"

        # Class labels
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

        # Load the model
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        # ----------- READ THE IMAGE AND PREPROCESSING -----------
        self.cap = cv2.VideoCapture(self.filepath)
        #cap = cv2.VideoCapture ('http://10.252.190.80:8080/video')
    Image_salida_upd=pyqtSignal(QImage)
    def run(self):
        self.hilo_corriendo=True
        count = 0
        while True:
            ret, frame, = self.cap.read()
            if ret == False:
                break
            #frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            
            frame_resized = cv2.resize(frame, (300, 200))

            # Create a blob
            blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 200), (127.5, 127.5, 127.5))
            #print("blob.shape:", blob.shape)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im3 = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ----------- DETECTIONS AND PREDICTIONS -----------
            self.net.setInput(blob)
            detections = self.net.forward()
            auxFrame = im3.copy()
            #print(detections)
            for detection in detections[0][0]:
                if self.classes[detection[1]]!="person":
                        continue
                
                if detection[2] > 0.45:
                    label = self.classes[detection[1]]
                    box = detection[3:7] * [width, height, width, height]
                    x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    body=auxFrame[y_start:y_end,x_start:x_end]
                    #print("cuerpo",body)
                    
                    
                    cv2.rectangle(im3, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    
                    cv2.imwrite(self.personPath +'/body_{}.png'.format(count),body)
                    count = count + 1
            frame =  imutils.resize(im3, width=1500, height=800)
            Image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            convertir_QT=QImage(Image.data,Image.shape[1],Image.shape[0],QImage.Format_RGB888)
            pic=convertir_QT.scaled(self.label_video.width(),self.label_video.height(),Qt.KeepAspectRatio)
            self.Image_salida_upd.emit(pic)
            if self.hilo_corriendo==False:
                break
        self.cap.release()
        cv2.destroyAllWindows()
    def stop(self):
        self.hilo_corriendo=False
        self.quit()