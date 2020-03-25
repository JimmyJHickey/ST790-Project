from base_gan import GAN
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import BinaryCrossentropy

class Vanilla_GAN(GAN):

    def __init__(self, g_gen=None):
        super().__init__(g_gen)

    def discriminator(self):
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(784,)))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

    def generator(self):
        """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
        and outputs images of size 28x28x1."""
        model = Sequential()
        model.add(Dense(1024, input_dim=100))
        model.add(LeakyReLU())
        model.add(Dense(128 * 7 * 7))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        if K.image_data_format() == 'channels_first':
            model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
            bn_axis = 1
        else:
            model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
            bn_axis = -1
        model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Convolution2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(axis=bn_axis))
        model.add(LeakyReLU())
        # Because we normalized training inputs to lie in the range [-1, 1],
        # the tanh function should be used for the output of the generator to ensure
        # its output also lies in this range.
        model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
        return model

    def discriminator_loss(self, real_output, generated_output):
        real_loss = BinaryCrossentropy(np.ones(real_output), real_output)
        generated_loss = BinaryCrossentropy(np.zeros(generated_output), generated_output)
        total_loss = real_loss + generated_loss
        return total_loss

    def generator_loss(self, generated_output):
        return BinaryCrossentropy(np.ones(generated_output), generated_output)

