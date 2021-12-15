import FaceDetect
import test
import keras
from mtcnn.mtcnn import MTCNN
import os

# Set MTCNN Tensorflow settings for local machine
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
faceDetector = MTCNN()

# setup face mask recognization model
maskDetector = keras.models.load_model("EfficientNetB3")

def main():
    FaceDetect.mtcnn_main(detector=faceDetector)
    test.test_main(model=maskDetector)

if __name__ == "__main__":
    main()