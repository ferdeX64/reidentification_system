import cv2
import imutils
from imutils.video import FPS
import os
import mediapipe as mp
import time
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ReidentificacionTextura(QThread):
    def __init__(self, label_video, filepath):
        super().__init__()
        self.label_video=label_video
        self.filepath=filepath
        mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        dataPath = 'Data_DNN' #Cambia a la ruta donde hayas almacenado Data
        self.imagePaths = os.listdir(dataPath)

        print('imagePaths=',self.imagePaths)
        # ----------- READ DNN MODEL -----------
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        # Leyendo el modelo
        self.face_recognizer.read('modeloDnn.xml')

        # Model architecture
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
        #cap = cv2.VideoCapture('pruebas/multitud.mp4')
        self.cap = cv2.VideoCapture(self.filepath)
        #cap = cv2.VideoCapture ('http://10.52.234.239:8080/video')
        #cap = cv2.VideoCapture ('http://10.252.15.251:8080/video')
        #todo el cuerpo imagen original bgr
    Image_salida_upd=pyqtSignal(QImage) 
    def run(self):
        self.hilo_corriendo=True
        with self.mp_pose.Pose(static_image_mode=False) as pose:
            VP=0
            VN=0
            FP=0
            FN=0
            inicio=time.time()
            fps=FPS().start()
            while True:
                ret, frame, = self.cap.read()
                #frame = cv2.resize(frame, (1700, 900))
                if ret == False: 
                        break
                height, width, _ = frame.shape
            
                frame_resized = cv2.resize(frame, (300, 300))
                # Create a blob
                blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
                #print("blob.shape:", blob.shape)
                #frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
                # ----------- DETECTIONS AND PREDICTIONS -----------
                self.net.setInput(blob)
                detections = self.net.forward()
                auxFrame = gray.copy()
                #print("AHJAHJHS",detections)
                for detection in detections[0][0]:
                        #print(classes[15])
                        if self.classes[detection[1]]!="person":
                            continue
                        if detection[2] > 0.45:
                            label = self.classes[detection[1]]
                            box = detection[3:7] * [width, height, width, height]
                            x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            #print("ver 2",box)
                            rostro=auxFrame[y_start:y_end,x_start:x_end]
                            print("rostro",rostro)
                            result = self.face_recognizer.predict(rostro)
                            salida='{:.2f}'.format(result[1])
                            print ("VECTOR",result)
                            if self.imagePaths[result[0]]=='Flor':
                                FP=FP+1
                            if "DESCONOCIDO"==self.imagePaths[result[0]]:
                                FN=FN+1
                            if result[1]<50:
                                VP=VP+1
                                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 4)
                                cv2.putText(frame,"%: {:.2f}".format(100-result[1])+" "+'{}'.format(self.imagePaths[result[0]]),(x_start, y_start - 25),2,1.2,(255,0,0),3,cv2.LINE_AA)
                                    #cv2.putText(frame, "%: {:.2f}".format(detection[2] * 100), (x_start+100, y_start -25), 1, 1.2, (255, 0, 0), 2)
                            else:
                                VN=VN+1
                                cv2.putText(frame,'Desconocido',(x_start, y_start - 25),2,1.1,(0,0,255),1,cv2.LINE_AA)
                                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
                frame =  imutils.resize(frame, width=1500)
                Image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                convertir_QT=QImage(Image.data,Image.shape[1],Image.shape[0],QImage.Format_RGB888)
                pic=convertir_QT.scaled(self.label_video.width(),self.label_video.height(),Qt.KeepAspectRatio)
                self.Image_salida_upd.emit(pic)
                #cv2.imshow("RE-IDENTIFICACION DE TEXTURA CAMARA 2", frame)
                if self.hilo_corriendo==False:
                        break
                fps.update()
            fin=time.time()
            final=fin-inicio
            fps.stop()
            print("Tiempo de reproducción: {:.2f}".format(fps.elapsed()))
            print("FPS aproximado: {:.2f}".format(fps.fps()))
            print("Tiempo de ejecución {:.2f}".format(final))
            print ("Verdaderos Positivos:",VP)
            print ("Verdaderos Negativos:",VN)
            print ("Falsos Positivos:",FP)
            print ("Falsos Negativos:",FN)
            self.cap.release()
            cv2.destroyAllWindows()
    def stop(self):
        self.hilo_corriendo=False
        self.quit() 