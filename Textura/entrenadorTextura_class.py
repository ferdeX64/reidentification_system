import cv2
import os
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class EntrenadorTextura(QThread):
    def __init__(self, button, button_end):
        super().__init__()
        self.button=button
        self.button_end=button_end
    def run(self):
        dataPath = 'Data_DNN' #Cambia a la ruta donde hayas almacenado Data
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
                print('Textura: ', nameDir + '/' + fileName)
                labels.append(label)
                facesData.append(cv2.imread(personPath+'/'+fileName,0)) #transformacion a escala de grises
            label = label + 1
        
        # Métodos para entrenar el reconocedor
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Entrenando el reconocedor de rostros
        self.button.setText("Entrenando...")
        print("Entrenando...")
        face_recognizer.train(facesData, np.array(labels))

        # Almacenando el modelo obtenido
        face_recognizer.write('modeloDnn.xml')
        self.button.setEnabled(True)
        self.button.setText("Modelo Almacenado - Entrenar de nuevo")
        self.button_end.setEnabled(True)
        print("Modelo almacenado...")

