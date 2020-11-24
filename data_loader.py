from PIL import Image
import numpy as np
import os
from random import random


# This is the data loader, can be modified
class data_generator(object):

    def __init__(self, im_size, loc, n, flip=True, suffix='png'):
        self.loc = "Datasets/" + loc
        self.flip = flip
        self.suffix = suffix
        self.n = n
        self.im_size = im_size
        self.images_list = self.read_image_list(self.loc)

    def get_batch(self, amount):

        idx = np.random.randint(0, self.n - 1, amount) + 1
        out = []

        for i in idx:
            temp = Image.open(self.images_list[i]).convert('RGB').resize((self.im_size, self.im_size))
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            if self.flip and random() > 0.5:
                temp1 = np.flip(temp1, 1)

            out.append(temp1)

        return np.array(out)

    def read_image_list(self, category):
        filenames = []
        print("list file")
        list = os.listdir(category)
        list.sort()
        for file in list:
            if 'jpg' in file:
                filenames.append(category + "/" + file)
        print("list file ending!")
        length = len(filenames)
        perm = np.arange(length)
        np.random.shuffle(perm)
        filenames = np.array(filenames)
        filenames = filenames[perm]

        return filenames

