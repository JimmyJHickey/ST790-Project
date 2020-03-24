# import tensorflow as tf
import numpy as np
# import keras


class GAN:

    def __init__(self, gen_input_shape, g_gen=None):
        """

        :param g_gen: (Optional) Wrapper function for G (generator) random input sample
            This will default to a Uniform [-1, 1] generator.
        """
        self.gen_input_shape = gen_input_shape
        # Set default random generator to uniform [-1, 1]
        if g_gen is None:
            self.g_gen = lambda s: np.random.uniform(-1.0, 1.0, size=s)
        else:
            self.g_gen = g_gen

    def g_sample(self):
        """
        g_sample
        Random sample input for generator.
        :param size: number of samples to generate
        :return: size randomly generated numbers
        """
        return self.g_gen(self.gen_input_shape)
