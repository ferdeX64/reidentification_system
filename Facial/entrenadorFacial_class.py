import cv2
import os
import numpy as np
import time
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class EntrenadorFacial(QThread):
    def __init__(self, button, button_end):
        super().__init__()
        self.button=button
        self.button_end=button_end
    def run(self):
        dataPath = 'Data' #Cambia a la ruta donde hayas almacenado Data
        peopleList = os.listdir(dataPath)
        print('Lista de personas: ', peopleList)
        self.button.setEnabled(False)
        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = dataPath + '/' + nameDir
            self.button.setText("Leyendo los datos...")
            print('Leyendo las imágenes')

            for fileName in os.listdir(personPath):
                print('Rostros: ', nameDir + '/' + fileName)
                labels.append(label)
                facesData.append(cv2.imread(personPath+'/'+fileName,0)) #transformacion a escala de grises
                image = cv2.imread(personPath+'/'+fileName,0)
               
            label = label + 1

        print('labels= ',labels)
        #print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
        #print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

        # Métodos para entrenar el reconocedor
        #face_recognizer = cv2.face.EigenFaceRecognizer_create()
        #face_recognizer = cv2.face.FisherFaceRecognizer_create()
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("VECTOR",face_recognizer)
        # Entrenando el reconocedor de rostros
        self.button.setText("Entrenando...")
        print("Entrenando...")
        inicio=time.time()
        face_recognizer.train(facesData, np.array(labels))
        tiempoEntrenamiento=time.time()-inicio
        print("Tiempo de entrenamiento: ",tiempoEntrenamiento)
        # Almacenando el modelo obtenido
        #face_recognizer.write('modeloEigenFace.xml')
        #face_recognizer.write('modeloFisherFace.xml')
        face_recognizer.write('modeloLBPHFace.xml')
        self.button.setEnabled(True)
        self.button.setText("Modelo Almacenado - Entrenar de nuevo")
        self.button_end.setEnabled(True)
        print("Modelo almacenado...")