import os

import cv2

import matplotlib.pyplot as plt


def load_image(infilename):
    img = cv2.imread(infilename)
    data = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    return data.reshape(-1, 128, 128, 3)


def predict(faceLoc,model):
    im = load_image(faceLoc)
    print(im.shape)
    res = model.predict(im).argmax(axis=1)
    return label[res[0]]


def plot_image(faceloc, destination, pred):
    fig, ax = plt.subplots()
    ax.imshow(plt.imread(faceloc))
    ax.set_title(f"PREDICTION:{pred}")
    plt.savefig(destination, bbox_inches='tight')
    plt.show()


def test_main(model):
    facesSubLoc = [f for f in os.listdir(facesLoc) if (not f.startswith('.')) and ("face" in f)]
    for filename in facesSubLoc:
        faceLoc = facesLoc + filename
        resultLoc = resultsLoc + filename
        res = predict(faceLoc,model)
        plot_image(faceLoc, resultLoc, res)


label = {0: 'mask_weared_incorrect', 1: 'with_mask', 2: 'without_mask'}
facesLoc = "Face_Detection_Result/"
resultsLoc = 'Face_Mask_Detection_Result/'

# predict("Face_Detection_Result/1014.png")
