import os
import shutil
import sys
from os import listdir

import cv2
import matplotlib.pyplot as plt
import openface

import utils

fileDir = os.path.dirname(os.path.realpath(__file__))

imgDim = 96
path = 'faces/'

align = openface.AlignDlib('dat/shape_predictor_68_face_landmarks.dat')
net = openface.TorchNeuralNet('dat/nn4.small2.v1.t7')


def get_representation(filename):
    bgr_img = cv2.imread(filename)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgb_img)
    if bb is None:
        print("Unable to find a face: {}".format(filename))
        return None

    aligned_face = align.align(imgDim, rgb_img, bb,
                               landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        print("Unable to align image: {}".format(filename))
        return None

    return net.forward(aligned_face)


def save_graph(features, filename):
    x = range(len(features))
    width = 1 / 1.5
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([-0.5, 0.5])
    plt.bar(x, features, width, color="blue")
    plt.ylabel('Probability')
    plt.savefig(filename)


def main(argv=None):
    for image_file in listdir(path):
        vector = get_representation(path + image_file)
        if vector is None:
            return

        user = utils.find_user(vector)
        if user is None:
            user = utils.create_user(vector)

        uid = user['uid']
        save_graph(vector, "hist_" + image_file)
        if not os.path.isdir(uid):
            os.makedirs(uid)

        shutil.copy(path + image_file, uid + "/" + image_file)

        print('User: ', user['uid'])


if __name__ == "__main__":
    sys.exit(main())
