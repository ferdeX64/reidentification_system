from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import cv2
from imutils import paths
import os
import cvzone
data = []
labels = []
numPoints=1
radius=4
def calculate_histogram(image, eps=1e-7):
        # Create a 2D array size of the input image
        
        lbp = feature.local_binary_pattern(image, numPoints,
                                          radius,
                                          method="uniform")
        # Make feature vector
        cv2.imshow("IMG",lbp)
        #Counts the number of time lbp prototypes appear
        (hist, _) = np.histogram(lbp.ravel(),
                                bins= np.arange(0, numPoints + 3),
                                range=(0, numPoints +2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        
        return hist
x_train = []
y_train = []
image_path = "img1/"
train_path = os.path.join(image_path, "train/")
test_path = os.path.join(image_path, "test/")

for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)
    print(folder_path)
    for file in os.listdir(folder_path):
        image_file = os.path.join(folder_path, file)
        image = cv2.imread(image_file)
        #image = cv2.resize(image,(300,300))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = calculate_histogram(gray)
        # Add the data to the data list
        y_train.append(image_file.split(os.path.sep)[-2])
        x_train.append(hist)
        # Add the label
model = LinearSVC(C=10, random_state=42)

model.fit(x_train, y_train)

for imagePath in paths.list_images(test_path):
    # load the image, convert it to grayscale, describe it,
    # # and classify it
    image = cv2.imread(imagePath)
    #image=cv2.resize(image,(300,300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = calculate_histogram(gray)
    prediction = model.predict(hist.reshape(1, -1))      
    #cv2.imshow("Image en escala de grises", gray)
    # display the image and the prediction
    print("persalida",prediction[0])
    print("persalida",hist)
    #cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
    imgStacked=cvzone.stackImages([image,gray],2,1)
    cv2.imshow("Image", imgStacked)
    cv2.waitKey(0)
    