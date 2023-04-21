import cv2
from multiprocessing import Process
import time
import os
import numpy
import mediapipe as mp#Importación de librerías
import imutils
import time
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from imutils.video import FPS
class ReconocimientoFacial(QThread):
    def __init__(self, label_video, filepath):
        super().__init__()
        self.label_video=label_video
        self.filepath=filepath
        self.dataPath = 'Data' #Nombre del dataset a listar
        self.imagePaths = os.listdir(self.dataPath)
        print('imagePaths=',self.imagePaths)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read('modeloLBPHFace.xml')# Lectura del modelo entrenado
        self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
        #cap = cv2.VideoCapture("pruebas/Alex3_cam1.mp4")#Lectura del video de entrada
        self.cap = cv2.VideoCapture(self.filepath)
        #cap = cv2.VideoCapture ('http://10.152.51.247:8080/video')#Lectura del video de entrada con una dirección IP
    Image_salida_upd=pyqtSignal(QImage)
    def run(self):
        self.hilo_corriendo=True
        VP=0
        VN=0 #Contadores para la matriz de confusión
        FP=0
        FN=0
        inicio=time.time()#Inicio de tiempo para calcular la velocidad del sistema
        self.fps=FPS().start()
        while True:
            ret,frame1 = self.cap.read()
            if ret == False: break#Condición si el video de entrada es valido
            #frame1 =  imutils.resize(frame1, width=800)
            frame1 = cv2.resize(frame1, (1200, 700), fx = 0, fy = 0,interpolation = cv2.INTER_CUBIC)#redimensionamiento del video de entrada
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)#Transformacion del video de entrada a escala de grises
            font=cv2.FONT_HERSHEY_SIMPLEX
            auxFrame1 = gray1.copy()
            faces = self.faceClassif.detectMultiScale(
                gray1,
                scaleFactor=1.2,
                minNeighbors=5,                    #Ajuste de los parametros del algoritmo de Viola Jones
                minSize=(1, 1),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x,y,w,h) in faces:#Ciclo para determinar la predicción del video de entrada 
                rostro = auxFrame1[y:y+h,x:x+w]#Definicion del cuadro delimitador
                rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)#Redimecionamiento de imagens de rostros del video de entrada
                result = self.face_recognizer.predict(rostro)#Calculo de la Predicción del modelo 
                salida='{:.2f}'.format(result[1])
                if self.imagePaths[result[0]]!=self.imagePaths[result[0]]:
                    FP=FP+1
                if "DESCONOCIDO"==self.imagePaths[result[0]]:
                    FN=FN+1
                if result[1] < 70:#condición del valor de confianza 
                    VP=VP+1
                    cv2.putText(frame1,"%: {:.2f}".format(100-result[1])+" "+'{}'.format(self.imagePaths[result[0]]),(x, y-25),1,1.2,(255,0,0),2,cv2.LINE_AA)
                    cv2.rectangle(frame1, (x,y),(x+w,y+h),(0,255,0),3)#Cuando la condición es verdadera
                else:
                    VN=VN+1
                    cv2.putText(frame1,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame1, (x,y),(x+w,y+h),(255, 0, 0),2)#Cuando la condición es falsa
            imgStacked=imutils.resize(frame1, width=1300)#Redimensionamiento del video de salida
            Image=cv2.cvtColor(imgStacked,cv2.COLOR_BGR2RGB)
            convertir_QT=QImage(Image.data,Image.shape[1],Image.shape[0],QImage.Format_RGB888)
            pic=convertir_QT.scaled(self.label_video.width(),self.label_video.height(),Qt.KeepAspectRatio)
            self.Image_salida_upd.emit(pic)
            ##cv2.imshow('RECONOCIMIENTO FACIAL CAMARA 1',imgStacked)#Salida del video de reconocimiento facial 
            if self.hilo_corriendo==False:
                break
            self.fps.update()
        fin=time.time()
        final=fin-inicio
        self.fps.stop()
        print("Tiempo de reproducción: {:.2f}".format(self.fps.elapsed()))
        print("FPS aproximado: {:.2f}".format(self.fps.fps()))
        print("Tiempo de ejecución {:.2f}".format(final))#Impresion de resultados 
        print ("Verdaderos Positivos:",VP)
        print ("Verdaderos Negativos:",VN)
        print ("Falsos Positivos:",FP)
        print ("Falsos Negativos:",FN)
        self.cap.release()
        cv2.destroyAllWindows()
    def stop(self):
        self.hilo_corriendo=False
        self.quit()        
