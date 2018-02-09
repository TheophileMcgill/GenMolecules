"""
Sequential variational autoencoder implementation
    Original model : 
        Variational auto-encoders for sentences by Bowman et al.
        https://arxiv.org/pdf/1511.06349.pdf
    Applied to chemical data :
        Automatic chemical design using a data-driven continuous 
          representation of molecules by Duvenaud et al.
        https://arxiv.org/pdf/1610.02415.pdf

    We learn a latent representation for the whole sequence and reconstruct
    the whole sequence back from it. 
    Our encoder is convolutional to take advantage of repetitive, translationally-invariant
    substrings that correspond to chemical substructures.
    Our decoder is recurrent to take advantage of recent advances in recurrent language models.

    We compute the loss as :
        Reconstruction error (as in every autoencoder)
            E[q(z|x)] where the expectation is taken over p(x|z)
            Computed as the average over a minibatch of the categorical cross
              entropy between input x and reconstruction x_hat
        Regularization term (KL divergence between posterior and prior)
            KL(q(z|x)||p(z))
            Computed in closed form (also averaged over a minibatch)
    """

import tensorflow as tf
import numpy as np

from keras import backend as K
from keras import objectives
from keras.layers import Input, Dense, Flatten, RepeatVector, TimeDistributed, Lambda
from keras.layers.convolutional import Convolution1D
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.optimizers import Adam


class SequentialVariationalAutoencoder2(object):

    # Convolutional encoder + sampling
    def _buildEncoder(self, x, n_time_steps, n_latent, eps_std=0.01):
        conv = Convolution1D(9, 9, activation='relu', name='conv1')(x)
        conv = Convolution1D(9, 9, activation='relu', name='conv2')(conv)
        conv = Convolution1D(10, 11, activation='relu', name='conv3')(conv)
        flat = Flatten(name='flatten')(conv)
        dense = Dense(435, name='dense_encode', activation='relu')(flat)

        z_mean = Dense(n_latent, activation='linear', name='z_mean')(dense)
        z_logvar = Dense(n_latent, activation='linear', name='z_logvar')(dense)

        def sampling(args):
            z_mean, z_logvar = args
            z_sigma = K.exp(0.5 * z_logvar)
            eps = K.random_normal(shape=(K.shape(z_logvar)), mean=0., std=eps_std)
            z = z_mean + z_sigma * eps # reparametrization trick
            return z

        z = Lambda(sampling, output_shape=(n_latent,), name='lambda')([z_mean, z_logvar])

        # Loss = KL divergence + Cross entropy
        def vae_loss(x, x_hat):
            x = K.flatten(x)
            x_hat = K.flatten(x_hat)
            CE = n_time_steps * objectives.binary_crossentropy(x, x_hat)
            KL = -0.5 * K.mean(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=1)
            return CE + KL

        return(vae_loss, z)

    # Recurrent decoder
    def _buildDecoder(self, z, n_time_steps, n_features, n_latent):
        function_of_z = Dense(n_latent, name='dense_decode', activation='relu')(z)
        repeat_vector = RepeatVector(n_time_steps, name='repeat_vector')(function_of_z)
        gru = GRU(501, return_sequences=True, name='gru1')(repeat_vector)
        gru = GRU(501, return_sequences=True, name='gru2')(gru)
        gru = GRU(501, return_sequences=True, name='gru3')(gru)
        x_hat = TimeDistributed(Dense(n_features, activation='softmax', name='x_hat'))(gru)
            
        return x_hat

    def __init__(self, n_time_steps, n_features, n_latent=292, weights_file=None):
        """ Construct the computational graph
            Inputs :
                n_time_steps : num of time steps per sequence
                n_features : num of entries in one hot vector at every time step
                n_latent : num of dimensions of latent representation
            """

        # Build encoder
        x = Input(shape=(n_time_steps, n_features))
        vae_loss, z = self._buildEncoder(x, n_time_steps, n_latent)
        self.encoder = Model(x, z)

        # Build autoencoder
        self.autoencoder = Model(
            x,
            self._buildDecoder(
                z,
                n_time_steps,
                n_features,
                n_latent))
        
        # Build decoder
        encoded_input = Input(shape=(n_latent,))
        self.decoder = Model(
            encoded_input, 
            self.autoencoder.layers[-1](
                self.autoencoder.layers[-2](
                    self.autoencoder.layers[-3](
                        self.autoencoder.layers[-4](
                            self.autoencoder.layers[-5](
                                self.autoencoder.layers[-6](
                                    encoded_input)))))))

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)

        # Compile autoencoder
        adam = Adam(lr=0.000005)
        self.autoencoder.compile(optimizer=adam, #optimizer='Adam',
                                 loss=vae_loss,
                                 metrics=['accuracy'])

    def save(self, filename):
        self.autoencoder.save_weights(filename)





