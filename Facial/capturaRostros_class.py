import cv2
import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class CapturaFacial(QThread):
    def __init__(self, name, filepath, label_video):
        super().__init__()
        self.personName = name
        self.filepath=filepath
        self.label_video=label_video
        dataPath = 'Data' 
        self.personPath = dataPath + '/' + self.personName

        if not os.path.exists(self.personPath):
            print('Carpeta creada: ',self.personPath)
            os.makedirs(self.personPath)

        self.cap = cv2.VideoCapture(self.filepath)
        #cap = cv2.VideoCapture('Antonio.mp4')
        #cap = cv2.VideoCapture ('http://192.168.137.94:8080/video')
    Image_salida_upd=pyqtSignal(QImage)
    def run(self):
        self.hilo_corriendo=True
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
        count = 112

        while True:
            ret, frame = self.cap.read()
            if ret == False: break
            frame = cv2.resize(frame, (1100, 800), fx = 0, fy = 0,interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()
            faces = faceClassif.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(1, 1),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                
                rostro = auxFrame[y:y+h,x:x+w]

                rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(self.personPath + '/rostro_{}.jpg'.format(count),rostro)
                count = count + 1
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