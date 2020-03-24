from base_gan import GAN
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy

class Vanilla_GAN(GAN):

    def __init__(self, g_gen=None):
        super().__init__(g_gen)

    def generator(self):
        

    def discriminator_loss(self, real_output, generated_output):
        real_loss = BinaryCrossentropy(np.ones(real_output), real_output)
        generated_loss = BinaryCrossentropy(np.zeros(generated_output), generated_output)
        total_loss = real_loss + generated_loss
        return total_loss

    def generator_loss(self, generated_output):
        return BinaryCrossentropy(ones(generated_output), generated_output)

