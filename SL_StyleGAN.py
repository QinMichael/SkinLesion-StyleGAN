import numpy as np
from functools import partial

# Imports for layers and models
from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from keras.models import model_from_json, Model
from keras.optimizers import Adam
import keras.backend as K

from AdaIN import AdaInstanceNormalization, InstanceNormalization


# Para setting
im_size = 256
latent_size = 100


# r1/r2 gradient penalty
def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                             axis=np.arange(1, len(gradients_sqr.shape)))

    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)


# Define the generator block
def g_block(inp, style, noise, fil, u=True):
    b = Dense(fil, kernel_initializer='he_normal', bias_initializer='zeros')(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer='he_normal', bias_initializer='ones')(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters=fil, kernel_size=1, padding='same', kernel_initializer='he_normal')(noise)

    if u:
        out = UpSampling2D(interpolation='bilinear')(inp)
        out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
    else:
        out = Activation('linear')(inp)

    if u:
        out = add([out, n])
        out = AdaInstanceNormalization()([out, b, g])
        out = LeakyReLU(0.01)(out)

    b = Dense(fil, kernel_initializer='he_normal', bias_initializer='zeros')(style)
    b = Reshape([1, 1, fil])(b)
    g = Dense(fil, kernel_initializer='he_normal', bias_initializer='ones')(style)
    g = Reshape([1, 1, fil])(g)

    n = Conv2D(filters=fil, kernel_size=1, padding='same', kernel_initializer='he_normal')(noise)

    out = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(out)
    out = add([out, n])
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(0.01)(out)

    return out


# Define the discriminator block
def d_block(inp, fil, p=True):
    temp = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(inp)
    temp = LeakyReLU(0.01)(temp)
    if p:
        temp = AveragePooling2D()(temp)
        temp = Conv2D(filters=fil, kernel_size=3, padding='same', kernel_initializer='he_normal')(temp)
    out = LeakyReLU(0.01)(temp)

    return out


# The Generator and Discriminator
class GAN(object):

    def __init__(self, lr=0.0001):

        # Models
        self.D = None
        self.G = None

        self.DM = None
        self.AM = None

        # Config
        self.LR = lr
        self.steps = 1

        # Init Models
        self.discriminator()
        self.generator()

    def discriminator(self):

        if self.D:
            return self.D

        inp = Input(shape=[im_size, im_size, 3])

        # Size
        x = d_block(inp, 16)  # Size / 2
        x = d_block(x, 32)  # Size / 4
        x = d_block(x, 64)  # Size / 8

        if im_size > 32:
            x = d_block(x, 128)  # Size / 16

        if im_size > 64:
            x = d_block(x, 192)  # Size / 32

        if im_size > 128:
            x = d_block(x, 256)  # Size / 64

        if im_size > 256:
            x = d_block(x, 384)  # Size / 128

        if im_size > 512:
            x = d_block(x, 512)  # Size / 256

        x = Flatten()(x)

        x = Dense(128)(x)
        x = Activation('relu')(x)

        x = Dropout(0.6)(x)
        x = Dense(1)(x)

        self.D = Model(inputs=inp, outputs=x)
        self.D.summary()

        return self.D

    def generator(self):

        if self.G:
            return self.G

        # 8 fully connected layers
        dense_units = 512
        inp_s = Input(shape=[latent_size])
        sty = Dense(dense_units, kernel_initializer='he_normal')(inp_s)
        sty = LeakyReLU(0.2)(sty)
        sty = Dense(dense_units, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.2)(sty)
        sty = Dense(dense_units, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.2)(sty)
        sty = Dense(dense_units, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.2)(sty)
        sty = Dense(dense_units, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.2)(sty)
        sty = Dense(dense_units, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.2)(sty)
        sty = Dense(dense_units, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.2)(sty)
        sty = Dense(dense_units, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.2)(sty)

        # Get the noise image and crop for each size
        inp_n = Input(shape=[im_size, im_size, 1])
        noi = [Activation('linear')(inp_n)]
        curr_size = im_size
        while curr_size > 4:
            curr_size = int(curr_size / 2)
            noi.append(Cropping2D(int(curr_size / 2))(noi[-1]))

        # Synthesis networks
        inp = Input(shape=[1])
        x = Dense(4 * 4 * dense_units, kernel_initializer='he_normal')(inp)
        x = Reshape([4, 4, dense_units])(x)
        x = g_block(x, sty, noi[-1], 512, u=False)

        if im_size >= 1024:
            x = g_block(x, sty, noi[7], 512)  # Size / 64
        if im_size >= 512:
            x = g_block(x, sty, noi[6], 384)  # Size / 64
        if im_size >= 256:
            x = g_block(x, sty, noi[5], 256)  # Size / 32
        if im_size >= 128:
            x = g_block(x, sty, noi[4], 192)  # Size / 16
        if im_size >= 64:
            x = g_block(x, sty, noi[3], 128)  # Size / 8

        x = g_block(x, sty, noi[2], 64)  # Size / 4
        x = g_block(x, sty, noi[1], 32)  # Size / 2
        x = g_block(x, sty, noi[0], 16)  # Size

        x = Conv2D(filters=3, kernel_size=1, padding='same', activation='sigmoid')(x)

        self.G = Model(inputs=[inp_s, inp_n, inp], outputs=x)
        self.G.summary()

        return self.G

    def AdModel(self):

        # D does not update
        self.D.trainable = False
        for layer in self.D.layers:
            layer.trainable = False

        # G does update
        self.G.trainable = True
        for layer in self.G.layers:
            layer.trainable = True

        # This model is simple sequential one with inputs and outputs
        gi = Input(shape=[latent_size])
        gi2 = Input(shape=[im_size, im_size, 1])
        gi3 = Input(shape=[1])

        gf = self.G([gi, gi2, gi3])
        df = self.D(gf)

        self.AM = Model(inputs=[gi, gi2, gi3], outputs=df)
        self.AM.summary()
        self.AM.compile(optimizer=Adam(self.LR, beta_1=0, beta_2=0.99, decay=0.00001), loss='mse')

        return self.AM

    def DisModel(self):

        # D does update
        self.D.trainable = True
        for layer in self.D.layers:
            layer.trainable = True

        # G does not update
        self.G.trainable = False
        for layer in self.G.layers:
            layer.trainable = False

        # Real Pipeline
        ri = Input(shape=[im_size, im_size, 3])
        dr = self.D(ri)

        # Fake Pipeline
        gi = Input(shape=[latent_size])
        gi2 = Input(shape=[im_size, im_size, 1])
        gi3 = Input(shape=[1])
        gf = self.G([gi, gi2, gi3])
        df = self.D(gf)

        # Samples for gradient penalty
        # For r1 use real samples (ri)
        # For r2 use fake samples (gf)
        da = self.D(ri)

        # Model With Inputs and Outputs
        self.DM = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, da])
        self.DM.summary()

        # Create partial of gradient penalty loss
        # For r1, averaged_samples = ri
        # For r2, averaged_samples = gf
        # Weight of 10 typically works
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=ri, weight=5)

        # Compile With Corresponding Loss Functions
        self.DM.compile(optimizer=Adam(self.LR, beta_1=0, beta_2=0.99, decay=0.00001),
                        loss=['mse', 'mse', partial_gp_loss])

        return self.DM
