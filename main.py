# face recognition part II
#IMPORT
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import tensorflow as tf
# Configurar TensorFlow para evitar advertencias
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Cargar modelo con manejo de advertencias de versión
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Intentar diferentes índices de cámara
cap = None
for camera_index in [0, 1, 2]:
    test_cap = cv.VideoCapture(camera_index)
    if test_cap.isOpened():
        ret, _ = test_cap.read()
        if ret:
            cap = test_cap
            print(f"Cámara encontrada en índice: {camera_index}")
            break
        test_cap.release()

if cap is None:
    print("No se pudo encontrar ninguna cámara disponible")
    exit()
# WHILE LOOP

while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)[0]
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' o ESC para salir
        break

cap.release()
cv.destroyAllWindows()