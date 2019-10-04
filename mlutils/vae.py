
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

class VAE:

    def __init__(self, original_dim, intermediate_dim, latent_dim):
        self.original_dim = original_dim
        self.input_shape = (original_dim, )
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

    def build(self):
        self.build_encoder()
        self.build_decoder()
        self.build_vae()

    def fit(self, x_train, x_test, batch_size, epochs):
        self.vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))

    def save_weights(self, fname='vae_mlp_mnist.h5', force_overwrite=True):
        import os
        if not force_overwrite and os.path.isfile(fname):
            raise ValueError("file %s exists")
        self.vae.save_weights(fname)

    def load_weights(self, fname='vae_mlp_mnist.h5'):
        self.vae.load_weights(fname)

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def build_encoder(self):
        # VAE model = encoder + decoder
        # build encoder model
        self.inputs = Input(shape=self.input_shape, name='encoder_input')
        self.xe = Dense(self.intermediate_dim, activation='relu')(self.inputs)
        self.z_mean = Dense(self.latent_dim, name='z_mean')(self.xe)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(self.xe)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')\
                       ([self.z_mean, self.z_log_var])

        # instantiate encoder model
        self.encoder = Model(self.inputs, 
                            [self.z_mean, self.z_log_var, self.z], 
                             name='encoder')

    def build_decoder(self):
        # build decoder model
        self.latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        self.xd = Dense(self.intermediate_dim, activation='relu')(self.latent_inputs)
        self.doutputs = Dense(self.original_dim, activation='sigmoid')(self.xd)

        # instantiate decoder model
        self.decoder = Model(self.latent_inputs, self.doutputs, name='decoder')
        

        
    def build_vae(self):
        self.outputs = self.decoder(self.encoder(self.inputs)[2])
        self.vae = Model(self.inputs, self.outputs, name='vae_mlp')
        self.reconstruction_loss = binary_crossentropy(self.inputs, self.outputs)
        self.reconstruction_loss *= self.original_dim
        self.kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
        self.kl_loss = K.sum(self.kl_loss, axis=-1)
        self.kl_loss *= -0.5
        self.vae_loss = K.mean(self.reconstruction_loss + self.kl_loss)
        self.vae.add_loss(self.vae_loss)
        self.vae.compile(optimizer='adam')


    def plot_results(self, x_test, y_test, batch_size=128):
        """Plots labels and MNIST digits as a function of the 2D latent vector

        # Arguments
            data (tuple): test data and label
            batch_size (int): prediction batch size
            model_name (string): which model is using this function
        """

        encoder, decoder = self.encoder, self.decoder

        # display a 2D plot of the digit classes in the latent space
        z_mean, _, _ = encoder.predict(x_test,batch_size=batch_size)
        plt.figure(figsize=(12, 10))
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
        plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.show()

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
