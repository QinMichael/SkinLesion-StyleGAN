from PIL import Image
from math import floor
import numpy as np
import time
import cv2 as cv

from keras.models import model_from_json, Model
from SL_StyleGAN import GAN
from AdaIN import AdaInstanceNormalization, InstanceNormalization
from data_loader import data_generator

# Para setting
im_size = 256
latent_size = 100
batch_size = 16
directory = "ISIC2018_mel"
n_images = 1113
suff = 'jpg'


# Style Z
def noise(n):
    return np.random.normal(0.0, 1.0, size=[n, latent_size])


# Noise Sample
def noise_image(n):
    return np.random.normal(0.0, 1.0, size=[n, im_size, im_size, 1])


class Train(object):

    def __init__(self, steps=-1, lr=0.0001, silent=True):

        self.GAN = GAN(lr=lr)
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.generator = self.GAN.generator()

        if steps >= 0:
            self.GAN.steps = steps

        self.lastblip = time.clock()

        self.noise_level = 0

        # load image data
        self.im = data_generator(im_size, directory, n_images, suffix=suff, flip=True)

        self.silent = silent

        self.ones = np.ones((batch_size, 1), dtype=np.float32)
        self.zeros = np.zeros((batch_size, 1), dtype=np.float32)
        self.nones = -self.ones

        self.enoise = noise(8)
        self.enoiseImage = noise_image(8)

    def train(self):

        # Train Alternating
        a = self.train_dis()
        b = self.train_gen()

        # Print information when training
        if self.GAN.steps % 20 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("G: " + str(b))
            s = round((time.clock() - self.lastblip) * 1000) / 1000
            print("T: " + str(s) + " sec")
            self.lastblip = time.clock()

            # Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000)+0)
            if self.GAN.steps % 500 == 0:
                self.evaluate(floor(self.GAN.steps / 100)+0)

        self.GAN.steps = self.GAN.steps + 1

    def train_dis(self):

        # Get Data
        train_data = [self.im.get_batch(batch_size), noise(batch_size), noise_image(batch_size), self.ones]

        # Train
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])

        return d_loss

    def train_gen(self):

        # Train
        g_loss = self.AdModel.train_on_batch([noise(batch_size), noise_image(batch_size), self.ones], self.zeros)

        return g_loss

    def evaluate(self, num=0):  # 8x4 images, bottom row is constant

        n = noise(32)
        n2 = noise_image(32)

        im2 = self.generator.predict([n, n2, np.ones([32, 1])])
        im3 = self.generator.predict([self.enoise, self.enoiseImage, np.ones([8, 1])])

        r12 = np.concatenate(im2[:8], axis=1)
        r22 = np.concatenate(im2[8:16], axis=1)
        r32 = np.concatenate(im2[16:24], axis=1)
        r43 = np.concatenate(im3[:8], axis=1)

        c1 = np.concatenate([r12, r22, r32, r43], axis=0)

        x = Image.fromarray(np.uint8(c1 * 255))

        x.save("Results/SL-StyleGAN_evaluated_" + str(num) + ".jpg")

    # Save Models
    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("SavedModels/" + name + ".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("SavedModels/" + name + "_" + str(num) + ".h5")

    # Load Models
    def loadModel(self, name, num):  # Load a Model

        file = open("SavedModels/" + name + ".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json, custom_objects={'AdaInstanceNormalization': AdaInstanceNormalization
                                                    })
        mod.load_weights("SavedModels/" + name + "_" + str(num) + ".h5")

        return mod

    # Save JSON and Weights into /SavedModels/
    def save(self, num):
        self.saveModel(self.GAN.G, "generator", num)
        self.saveModel(self.GAN.D, "discriminator", num)

    # Load JSON and Weights from /SavedModels/
    def load(self, num):  # Load JSON and Weights from /Models/
        steps1 = self.GAN.steps

        self.GAN = None
        self.GAN = GAN()

        # Load Models
        self.GAN.G = self.loadModel("generator", num)
        self.GAN.D = self.loadModel("discriminator", num)

        self.GAN.steps = steps1

        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()

    def generate_images(self):
        num_img = 500  # num of images to generate
        rnd_n = np.random.RandomState(6)
        n = rnd_n.normal(0.0, 1.0, size=[num_img, latent_size])
        # n[:, 1] = 2.5
        rnd_img = np.random.RandomState(8)
        n_img = rnd_img.normal(0.0, 1.0, size=[num_img, im_size, im_size, 1])

        gen_imgs = self.generator.predict([n, n_img, np.ones([num_img, 1])])

        for k in range(num_img):
            gen_imgs[k] = cv.normalize(gen_imgs[k], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            (b, g, r) = cv.split(gen_imgs[k])
            gen_imgs[k] = cv.merge([r, g, b])
            cv.imwrite("Results/SL-StyleGAN_generated_%d.jpg" % (k + 1), gen_imgs[k])


if __name__ == "__main__":
    epoch = 20000

    model = Train(lr=0.0001, silent=False)        # lr = 0.001 bad results

    # This is for the model loading and evaluating, or continue training
    # model.load(2)

    for i in range(epoch):
        model.train()

    # Generate images
    model.generate_images()
