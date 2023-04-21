import cv2
import os
import numpy as np


def train():
	dataPath = 'Data_Color' #Cambia a la ruta donde hayas almacenado Data
	peopleList = os.listdir(dataPath)
	print('Lista de personas: ', peopleList)

	labels = []
	facesData = []
	label = 0
	for nameDir in peopleList:
		personPath = dataPath + '/' + nameDir
		print('Leyendo las imágenes')

		for fileName in os.listdir(personPath):
			print('Textura: ', nameDir + '/' + fileName)
			labels.append(label)
			facesData.append(cv2.imread(personPath+'/'+fileName,0)) #transformacion a escala de grises
		label = label + 1
    
	# Métodos para entrenar el reconocedor
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()

	# Entrenando el reconocedor de rostros
	print("Entrenando...")
	face_recognizer.train(facesData, np.array(labels))

	# Almacenando el modelo obtenido
	face_recognizer.write('modeloColor.xml')
	print("Modelo almacenado...")

def main():
	train()

if __name__ == "__main__":
	main()
