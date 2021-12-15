# from imutils.video import VideoStream
# import imutils
import os

import keras
from cv2 import cv2
from mtcnn.mtcnn import MTCNN

# Set MTCNN Tensorflow settings for local machine
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
faceDetector = MTCNN()
print("MTCNN successfully loaded")
# setup face mask recognition model
maskDetector = keras.models.load_model("EfficientNetB3")
print("maskDetector successfully loaded")


def rescale_frame(img, percent=75):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def rescale_facebox_info(faceinfo):
    return [int((100 / pct) * num) for num in faceinfo]


def draw_box(img, face_infos):
    for face_info in face_infos:
        x_start, y_start, width, height = rescale_facebox_info(face_info['box'])
        x_end, y_end = x_start + width, y_start + height
        # extract the face
        face = img[y_start:y_end, x_start:x_end]

        # predict mask
        data = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        maskresult = predict_mask(data.reshape(-1, 128, 128, 3), maskDetector)

        # draw result
        color = (0, 255, 0) if maskresult == 1 else (0, 0, 255)
        cv2.putText(img, label[maskresult], (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color, 2)


def predict_mask(im, model):
    res = model.predict(im).argmax(axis=1)
    return res[0]


label = {0: 'mask_weared_incorrect', 1: 'with_mask', 2: 'without_mask'}
pct = 10  # percentage
# define a video capture object
vid = cv2.VideoCapture(1)

while True:
    # Capture the video img
    # by img
    ret, frame = vid.read()
    # due to the limitation of local machien, we downsampled img
    frame10 = rescale_frame(frame, pct)
    faceinfos = faceDetector.detect_faces(frame10)

    draw_box(frame, faceinfos)

    # Display the resulting img
    cv2.imshow('img', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
