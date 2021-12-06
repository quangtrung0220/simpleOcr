#
# You can modify this files
#
import os
import random
import numpy as np
import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt

from collections import namedtuple

from sklearn.metrics import classification_report


class HoadonOCR():

    def __init__(self):
        # Init parameters, load model here
        self.model_test = torchvision.models.densenet121(pretrained=False, progress=False)
        self.model_test.classifier = torch.nn.modules.linear.Linear(in_features=1024, out_features=4, bias=True)
        self.model_test.load_state_dict(torch.load('./desnet121.pth', map_location=torch.device('cpu')))

        self.labels = ['highlands', 'others',  'phuclong', 'starbucks']

    # TODO: implement find label
    def find_label(self, img):

        # pre processing
        img = self.preprocess(img)

        # convert to PiL image and then covert to tensor to use deep learning
        # resize to size 224x224
        toPIL = transforms.ToPILImage()
        resize_224 = transforms.Resize((224, 224))
        tensor = transforms.ToTensor()
        # norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_test_224 = transforms.Compose([toPIL, resize_224, tensor])
        img = transform_test_224(img)
        img = img.view(1, 3, 224, 224)

        # Find label: get index of max value in output array
        with torch.no_grad():
            self.model_test.eval()

            output = self.model_test(img)
            ypred = int(torch.max(output.data, 1)[1].numpy())

        return self.labels[ypred]

    def preprocess(self, img):

        # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # using histogram equalize
        gray = cv2.equalizeHist(gray)

        # Get rid of noise with Gaussian Blur filter
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect white regions
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))

        # Correct the nonuniform illumination of the background and convert gray image to binary image
        bg = cv2.morphologyEx(blurred, cv2.MORPH_DILATE, rectKernel)
        out_gray = cv2.divide(blurred, bg, scale=255)
        out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

        # find contours
        contours, hierarchy = cv2.findContours(out_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get largest contour
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        # image_with_largest_contours = cv2.drawContours(img.copy(), largest_contours, -1, (0, 255, 0), 3)

        # crop image by largest contour
        for cnt in largest_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 50:
                break

        cropped = img[y:y + h, x:x + w]

        # cv2.imshow("test", cropped)
        # cv2.waitKey()
        return cropped
