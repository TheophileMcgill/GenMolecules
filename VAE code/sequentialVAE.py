"""
Sequential variational autoencoder implementation
    Author : Theophile Gervet
    Original paper : A Recurrent Latent Variable Model for Sequential Data
                     https://arxiv.org/pdf/1506.02216v3.pdf
                     by Chung et al.

    We follow closely the notation from the paper :
        x_t : input at time step t
        z_t : latent reprensentation at time step t
        h_t : recurrent layer state at time step t

    The model is an enrolled RNN with a VAE at every time step.
    At every time step t we compute :
        Prior :      p(z_t|h_t-1)
        Inference :  q(z_t|h_t-1,x_t)   (encoder at time step t)
        Generation : p(x_t|h_t-1,z_t)   (decoder at time step t)
        Recurrence : h_t depends on h_t-1, x_t, z_t

    We compute the loss as the sum over all time steps t of :
        Reconstruction error (as in every autoencoder)
            E[q(z_t|h_t-1,x_t)] where the expectation is taken over p(x_t|h_t-1,z_t)
            Computed as the average over a minibatch of the categorical cross entropy
             between input x_t and reconstruction x_hat_t
        Regularization term (KL divergence between posterior and prior)
            KL(q(z_t|h_t-1,x_t)||p(z_t))
            Computed in closed form (also averaged over a minibatch)
"""

import numpy as np
import tensorflow as tf
from keras.layers import Dense, merge, Flatten

class SequentialVariationalAutoencoder(object):

    def compute_prior(self):
        """ Compute parameters of prior distribution p(z_t|h_t-1)
                Note that self.h = h_t-1

                Outputs :
                    mean : mean of prior
                    logvar : log of variance of prior (enforce logvar >= 0 (=) var >= 1)
            """
        hidden = Dense(self.n_hidden_units, activation='relu')(self.h)
        hidden = Dense(self.n_hidden_units, activation='relu')(hidden)
        mean = Dense(self.n_latent)(hidden)
        logvar = Dense(self.n_latent, activation='relu')(hidden)

        return mean, logvar

    def compute_posterior(self, x_t):
        """ Compute posterior q(z_t|x_t, h_t-1)

                Outputs :
                    z : sample from posterior
                    mean : mean of posterior
                    logvar : log of variance of posterior (no need to enforce logvar >= 0)
            """
        # Merge inputs
        inputs = merge([self.h, x_t], mode='concat')

        hidden = Dense(self.n_hidden_units, activation='relu')(inputs)
        hidden = Dense(self.n_hidden_units, activation='relu')(hidden)
        mean = Dense(self.n_latent)(hidden)
        logvar = Dense(self.n_latent)(hidden)

        # Sample from posterior
        sigma = tf.exp(0.5 * logvar)
        epsilon = tf.random_normal(tf.shape(logvar)) # sample from Gaussian with Id covar
        z = mean + sigma * epsilon # reparametrization trick

        return z, mean, logvar

    def compute_reconstruction(self, z_t):
        """ Compute posterior p(x_t|z_t, h_t-1)

                Outputs :
                    x_hat : unnormalized reconstruction
                            i.e if x_t is a one hot vector [0, 1, 0]
                             x_hat at time step t might be [3, 9, 2]
            """
        #  Merge inputs
        inputs = merge([self.h, z_t], mode='concat')

        hidden = Dense(self.n_hidden_units, activation='relu')(inputs)
        hidden = Dense(self.n_hidden_units, activation='relu')(hidden)
        x_hat = Dense(self.n_features)(hidden)
        
        return x_hat

    def compute_KL(self, prior_mean, prior_logvar, post_mean, post_logvar):
        """ Compute KL divergence between prior p(z_t|h_t-1) and posterior q(z_t|x_t, h_t-1)
            at time step t in closed form
            """
        prior_var, post_var = tf.exp(prior_logvar), tf.exp(post_logvar)
        KLD = 0.5 * tf.reduce_sum(prior_logvar - post_logvar +
            (-1 + post_var + (post_mean - prior_mean) ** 2) / prior_var,
            axis=1)

        return KLD

        
    def __init__(
        self, n_time_steps, n_features, batch_size,
        n_latent, n_hidden_units, n_rec_units, n_rec_layers, 
        optimizer=tf.train.AdamOptimizer(0.001), restore=None):
        """ Constructs the computational graph by enrolling a RNN and constructing a 
            variational autoencoder at every time step t.
            At every time step, we need h_t-1 to compute z_t and x_hat_t, hence we need 
            to enroll the RNN. This means we cannot use Keras builtin recurrent layers, 
            we need to use Tensorflow.

            Inputs :
                n_time_steps : num of time steps per sequence
                n_features : num of entries in one hot vector encoding input at every time step
                batch_size : size of batch of data (necessary to initialize hidden state)
                n_latent : num of dimensions of latent representation at every time step
                n_hidden_units : number of hidden units in every fully connected hidden layer
                n_rec_units : size of recurrent layers
                n_rec_layers : number of recurrent layers
                optimizer : tensorflow optimizer
                restore : 
                    if None initialize new weights
                    if "path" restore weights from file at path
            """

        self.n_features = n_features
        self.n_latent = n_latent
        self.n_hidden_units = n_hidden_units

         # Placeholder for input
        input_shape = [batch_size, n_time_steps, n_features]
        self.x = tf.placeholder(tf.float32, shape=input_shape)

        # Variables that will be updated or filled in at each time step while enrolling
        self.CE = tf.Variable(0.)   # cross entropy term of loss function
        self.KLD = tf.Variable(0.)  # KL divergence term of loss function
        x_pred = []   # will hold reconstruction for x at every time step t
        z = []        # will hold sample from posterior latent distribution at every time step t
        
        # Build recurrent layer
        gru = tf.contrib.rnn.GRUCell(n_rec_units)
        stacked_gru = tf.contrib.rnn.MultiRNNCell([gru] * n_rec_layers)

        # Initialize recurrent state
        # states holds states of all recurrent layers in a tuple
        states = stacked_gru.zero_state(batch_size, tf.float32)
        # self.h holds state of last recurrent layer
        self.h = states[-1]

        # Enroll recurrent layer
        with tf.variable_scope("enrolling") as scope:
            for t in range(n_time_steps):
                print('Enrolling graph, time step : ' + str(t))

                # First iteration creates variables that represent lstm parameters
                # Every subsequent iterations look them up in the scope by name
                if t > 0:
                    scope.reuse_variables()

                # Compute prior p(z_t|h_t-1)
                prior_mean, prior_logvar = self.compute_prior()

                # Compute posterior q(z_t|x_t, h_t-1)
                x_t = self.x[:,t,:]
                z_t, post_mean, post_logvar = self.compute_posterior(x_t)
                z.append(z_t) # keep result

                # Compute reconstruction p(x_t|z_t, h_t-1)
                x_hat_t = self.compute_reconstruction(z_t)

                # Compute one hot output from this reconstruction
                x_pred_t = tf.one_hot(tf.argmax(x_hat_t, axis=1), depth = self.n_features)
                x_pred.append(x_pred_t) # keep result

                # Update recurrent state h_t given z_t, x_t, h_t-1
                inputs = merge([x_t, z_t], mode='concat')
                _, states = stacked_gru(inputs, states)
                self.h = states[-1]

                # Compute KL divergence term of objective function
                KLD = self.compute_KL(prior_mean, prior_logvar, post_mean, post_logvar)
                # Compute cross entropy term of objective function
                CE = tf.nn.softmax_cross_entropy_with_logits(labels=x_t, logits=x_hat_t)
                # Update components of loss (averaged over minibatch)
                self.KLD += tf.reduce_mean(KLD)
                self.CE += tf.reduce_mean(CE)


        # Convert lists gathered during enrolling to tensors
        self.x_pred = tf.stack(x_pred, axis=1)
        self.z = tf.stack(z, axis=1)

        # Loss = KLD + CE
        self.loss = self.CE + self.KLD

        # Training step
        self.train_step = optimizer.minimize(self.loss)

        # Initialize session to run the graph
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        if restore is None: # Initialize new weights
            self.sess.run(tf.global_variables_initializer())
        else: # Restore weights from path
            self.saver.restore(self.sess, restore)


    def partial_fit(self, X):
        _, loss, KLD, CE = self.sess.run(
            [self.train_step, self.loss, self.KLD, self.CE], feed_dict={self.x: X})

        return (loss, KLD, CE)

    def reconstruct(self, X):
        return self.sess.run(self.x_pred, feed_dict={self.x: X})

    def generate(self, Z=None):
        # TODO : how to get prior without input x ?
        # if no input, sample latent representation from prior
        return self.sess.run(self.x_pred, feed_dict={self.z: Z})
    
    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def calc_loss(self, X):
        return self.sess.run(self.loss, feed_dict = {self.x: X})

    def save(self, path):
        self.saver.save(self.sess, path)
    
