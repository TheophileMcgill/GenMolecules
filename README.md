# GeneratingMolecules

Finding molecules with desirable properties is critical for many fields related to chemical engineering. However, this task is not at all trivial, as optimization in molecular space is difficult. 

In this work, we explore two generative approaches involving deep learning introduced in previous literature. The first approach consists in training a recurrent neural network on a huge set of molecules (represented as strings) and then fine tune on a smaller set with desired characteristics. The second approach consists in training a deep variational autoencoder to embed molecules in a continuous latent space in order to interpolate between molecules and optimize for characteristics in this more convenient space.
