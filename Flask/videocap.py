from statistics import mode

import cv2
from keras.models import load_model
import numpy as np


detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = ('trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5')

emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                         4: 'sad', 5: 'surprise', 6: 'neutral'}

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)


face_detection = cv2.CascadeClassifier(detection_model_path)


emotion_classifier = load_model(emotion_model_path)


emotion_target_size = emotion_classifier.input_shape[1:3]


emotion_window = []


cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
emotion_text = ""


tempbool= True
while tempbool:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray_image, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray_image[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(64,64),interpolation=cv2.INTER_AREA)


        gray_face = roi_gray.astype('float')/255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        #tempbool = False

    cv2.imshow('window_frame', bgr_image)
    print(emotion_text)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('quit')
        break
video_capture.release()
cv2.destroyAllWindows()

def meth():
    return emotion_text
