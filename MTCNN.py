import os

import matplotlib.pyplot as plt
from PIL import Image
from mtcnn.mtcnn import MTCNN


def draw_facebox(img, result_list, show_plot):
    # plot the image
    plt.imshow(img)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = plt.Rectangle((x, y), width, height, fill=False, color='orange')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = plt.Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    if show_plot:
        plt.show()
    return plt


def read_img(filepath, show_plot):
    pixels = plt.imread(filepath)
    print("Shape of image/array:", pixels.shape)
    plt.imshow(pixels)

    if show_plot:
        plt.show()
    return pixels


def crop_save_face(pixels, results, folder_result, filename):
    for i, face in enumerate(results):
        # extract the bounding box from the first face
        x1, y1, width, height = face['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize((160, 160))
        image.save(folder_result + 'face ' + str(i) + ' ' + filename, bbox_inches='tight')


def mtcnn_main(show_plot=True):
    # data location
    folder_data = 'Face_Detection_Data/'
    folder_result = 'Face_Detection_Result/'

    # Set MTCNN Tensorflow settings for local machine
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    detector = MTCNN()

    for file in os.listdir(folder_data):
        if not file.startswith('.'):  # exclude hidden files
            # get filename, filepath, file extension
            filename, file_extension = os.path.splitext(file)
            filepath = folder_data + filename + file_extension

            # detect faces
            pixels = read_img(filepath, show_plot)
            results = detector.detect_faces(pixels)

            # draw and save faces
            plot = draw_facebox(pixels, results, show_plot)
            plot.savefig(folder_result + filename, bbox_inches='tight')

            # save each face
            crop_save_face(pixels, results, folder_result, file)
