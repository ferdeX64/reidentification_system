import cv2
from multiprocessing import Process
import time
import os
import numpy
import mediapipe as mp#Importación de librerías
import imutils
import time
from imutils.video import FPS
dataPath = 'Data' #Nombre del dataset a listar
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modeloLBPHFace.xml')# Lectura del modelo entrenado
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
#cap = cv2.VideoCapture("pruebas/Alex3_cam1.mp4")#Lectura del video de entrada
cap = cv2.VideoCapture("pruebas/Alex1_cam1.mp4")
#cap = cv2.VideoCapture ('http://10.152.51.247:8080/video')#Lectura del video de entrada con una dirección IP
def Facial():
    VP=0
    VN=0 #Contadores para la matriz de confusión
    FP=0
    FN=0
    inicio=time.time()#Inicio de tiempo para calcular la velocidad del sistema
    fps=FPS().start()
    while True:
        ret,frame1 = cap.read()
        if ret == False: break#Condición si el video de entrada es valido
        #frame1 =  imutils.resize(frame1, width=800)
        frame1 = cv2.resize(frame1, (1200, 700), fx = 0, fy = 0,interpolation = cv2.INTER_CUBIC)#redimensionamiento del video de entrada
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)#Transformacion del video de entrada a escala de grises
        font=cv2.FONT_HERSHEY_SIMPLEX
        auxFrame1 = gray1.copy()
        faces = faceClassif.detectMultiScale(
            gray1,
            scaleFactor=1.2,
            minNeighbors=5,                    #Ajuste de los parametros del algoritmo de Viola Jones
            minSize=(1, 1),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x,y,w,h) in faces:#Ciclo para determinar la predicción del video de entrada 
            rostro = auxFrame1[y:y+h,x:x+w]#Definicion del cuadro delimitador
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)#Redimecionamiento de imagens de rostros del video de entrada
            result = face_recognizer.predict(rostro)#Calculo de la Predicción del modelo 
            salida='{:.2f}'.format(result[1])
            if imagePaths[result[0]]!=imagePaths[result[0]]:
               FP=FP+1
            if "DESCONOCIDO"==imagePaths[result[0]]:
               FN=FN+1
            if result[1] < 70:#condición del valor de confianza 
                VP=VP+1
                cv2.putText(frame1,"%: {:.2f}".format(100-result[1])+" "+'{}'.format(imagePaths[result[0]]),(x, y-25),1,1.2,(255,0,0),2,cv2.LINE_AA)
                cv2.rectangle(frame1, (x,y),(x+w,y+h),(0,255,0),3)#Cuando la condición es verdadera
            else:
                VN=VN+1
                cv2.putText(frame1,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame1, (x,y),(x+w,y+h),(255, 0, 0),2)#Cuando la condición es falsa
        imgStacked=imutils.resize(frame1, width=1300)#Redimensionamiento del video de salida
        cv2.imshow('RECONOCIMIENTO FACIAL CAMARA 1',imgStacked)#Salida del video de reconocimiento facial 
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
        fps.update()
    fin=time.time()
    final=fin-inicio
    fps.stop()
    print("Tiempo de reproducción: {:.2f}".format(fps.elapsed()))
    print("FPS aproximado: {:.2f}".format(fps.fps()))
    print("Tiempo de ejecución {:.2f}".format(final))#Impresion de resultados 
    print ("Verdaderos Positivos:",VP)
    print ("Verdaderos Negativos:",VN)
    print ("Falsos Positivos:",FP)
    print ("Falsos Negativos:",FN)
    cap.release()
    cv2.destroyAllWindows()
    
