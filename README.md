# Real-Time Facial Mask Detection

## Abstract
This project aims to create a backend program for real-time facial mask detection in both static images and webcam video streams. The process involves recognizing faces in images or video frames, cropping the faces, and then determining whether the individuals are wearing facial masks. The face detection utilizes the MTCNN algorithm, while various deep learning models are implemented for mask detection.

##  Introduction
The project focuses on constructing a backend program for real-time facial mask detection in video streams. The process involves two main steps: face recognition in video frames and determining if the recognized faces are wearing masks. MTCNN is chosen for face recognition, and multiple deep learning algorithms are implemented for mask detection using the Face Mask Detection dataset from Kaggle.

##  Face Detection
To detect facial masks, a model to detect faces must be constructed. The MTCNN algorithm is chosen for face detection, alignment, cropping, and scaling. MTCNN comprises three sub-networks: Proposal Network (PNet), Refine Network (RNet), and Output Network (ONet). The downsized input images are used to speed up the pipeline.

### MTCNN
MTCNN is a deep learning algorithm for face detection and alignment. It consists of three sub-networks: PNet, RNet, and ONet. The algorithm efficiently processes face images from rough to detailed.

### Result
After implementing MTCNN face detection, successful face detection results are obtained. The faces are cropped and resized for mask detection.

##  Mask Detection
For mask detection, Keras is used to build computer vision deep learning algorithms, including Convolutional Neural Networks (CNNs), ResNets, InceptionNets, DenseNets, and EfficientNets. The models are compared based on accuracy, loss, and training time.

### Methods
Different architectures are used for mask recognition, including CNNs, ResNets, InceptionNets, DenseNets, and EfficientNets.

#### CNNs
Three CNN models are implemented, with the first model achieving the best accuracy on the test set.

#### ResNets
Various ResNet architectures are explored, with ResNet50 achieving the highest accuracy on the test set.

#### InceptionNets
Inception ResNet architectures are implemented, with Inception ResNet achieving high accuracy with faster training time.

#### DenseNets
DenseNet architectures are used. DenseNet121 achieves the highest accuracy on the test set.

#### EfficientNets
EfficientNet architectures are implemented, with EfficientNetB0 achieving the highest accuracy on the test set.

### Comparison and Discussion
The models are compared based on accuracy, loss, and training time. The best-performing model, EfficientNetB0, is chosen for mask detection in static images and webcam video streams.

##  Final Result and Analysis
The best-performing mask detection model, EfficientNetB0 is applied to construct a pipeline for detecting facial masks in static images and webcam video streams. The detection results are analyzed, and limitations, such as lower accuracy in low-resolution images, are identified for future improvements.

