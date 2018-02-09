import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from random import shuffle
from PIL import Image

from sequentialVAE2 import SequentialVariationalAutoencoder2

from sklearn.decomposition import PCA
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
#from rdkit.Contrib import SA_Score
from rdkit.Chem import Draw

MAX_LEN = 120
TRAIN_FILE = 'data/250k_rndm_zinc_drugs_clean.smi'
LIMIT = None
BATCH_SIZE = 300 # maximum batch size fitting in gpu
NUM_EPOCHS = 40
TRAIN = False # set to true to train model


""" Filter large molecules and pad sequences at the end
	"""
def filter_and_pad(string):
	if len(string) <= MAX_LEN:
		return string + " " * (MAX_LEN - len(string))


""" Convert output of VAE back to smiles
	Inputs : 
		X : numpy array in the format used by our variational autoencoder
		index_to_char : dictionary 
	Output : 
		list of smiles strings
	"""
def one_hot_to_smiles(X, index_to_char):
	mols = [X[i].tolist() for i in range(X.shape[0])]
	mols = [(''.join([index_to_char[np.argmax(t)] for t in mol])).strip() for mol in mols]
	return mols


""" Draw molecule from smiles
	Inputs :
		smi : smiles representation of molecule
		filename : path of the file where to draw
	"""
def draw_molecule(smi, filename):
	mol = Chem.MolFromSmiles(smi)
	if mol is not None:
		Draw.MolToFile(mol, filename)


"""	Compute reconstruction accuracy
	Inputs :
		vae : keras model 
		X : training set 
		index_to_char : dictionary used for conversion
		lim : set to None to consider whole training set
		verbose : set to true to print out originals vs reconstructions
		draw : set to true to draw reconstructed molecules to png files
	Output :
		reconstruction accuracy
	"""
def compute_reconstruction_accuracy(vae, X, index_to_char, lim=10, verbose=False, draw=False):
	if lim is not None:
		X = X[:lim]
	# Reconstruct using autoencoder
	X_hat = vae.autoencoder.predict(X)
	# Convert back to smiles
	mols = one_hot_to_smiles(X, index_to_char)
	mols_hat = one_hot_to_smiles(X_hat, index_to_char)
	# Compute reconstruction accuracy
	n_matches = 0
	for i in range(X.shape[0]):
		if mols[i] == mols_hat[i]:
			n_matches += 1
		if verbose:
			print("Original {}".format(mols[i]))
			print("Reconstr {}".format(mols_hat[i]))
		if draw:
			draw_molecule(mols_hat[i], 'draw/mol{}.png'.format(i))
	return float(n_matches)/X.shape[0]


"""	Plot projection of latent space in two dimensions
	Inputs :
		vae : keras model 
		X : training set
		y : list of labels used for coloring
		verbose : set to true to print ratio of variance explained by first two
					eigenvalues
	"""
def plot_latent(vae, X, y=None, verbose=False):
	# Project into latent space
	Z = vae.encoder.predict(X)
	# Project latent space to two dimensions using PCA
	pca = PCA(n_components=2)
	Z_proj = pca.fit_transform(Z)
	if verbose:
		print("Explained variance ratio {}".format(pca.explained_variance_ratio_))
	# Plot projection
	plt.figure()
	if y != None:
		y = np.array(y)
		y = (y-np.mean(y))/np.std(y) # Standardize to zero mean, unit variance
		#plt.scatter(Z_proj[:,0], Z_proj[:,1], c=y, s=1, lw=0)
		plt.scatter(Z[:,45], Z[:,46], c=y, s=1, lw=0)
		plt.colorbar()
	else:
		plt.figure()
		plt.scatter(Z_proj[:,0], Z_proj[:,1], s=10, facecolor='0.5', lw=0)
	#plt.savefig('plots_noncanonical/latent.png')
	plt.show()


"""	Plot all latent dimensions overlaid
	Inputs :
		vae : keras model 
		X : training set
	"""
def plot_latent_with_prior(vae, X, y=None, verbose=False):
	# Project into latent space
	Z = vae.encoder.predict(X)
	mu, sigma = 0, 0.1 # Latent prior parameters for all 292 dimensions
	# Plot projection
	fig = plt.figure()
	for i in range(292):
		ax = fig.add_subplot(111)
		n, bins, patches = ax.hist(Z[:,i], 50, normed=1, facecolor='green', alpha=0.75)
		bincenters = 0.5*(bins[1:]+bins[:-1])
		y = mlab.normpdf(bincenters, mu, sigma)
		if i == 0:
			l = ax.plot(bincenters, y, 'r', linewidth=1, label='Gaussian prior')
		else:
			l = ax.plot(bincenters, y, 'r', linewidth=1)
	ax.set_xlabel('Value in latent space along one dimension')
	ax.set_ylabel('Density')
	ax.set_title('All 292 latent dimensions overlaid')
	ax.legend()
	ax.set_xlim(-0.6, 0.6)
	ax.grid(True)
	plt.savefig('plots_latent/all_latent_dimensions.png')
	plt.show()



"""	Interpolate between two molecules in latent space
	Inputs :
		X : numpy array containing two molecules in one hot format
		n_steps : number of steps
	Ouput : list of smiles molecules decoded from interpolated values
			between src and dest in order
	"""
def interpolate(X, n_chars, char_to_index, index_to_char, vae, n_steps=10000):
	# Project source and destination into latent space
	Z = vae.encoder.predict(X)
	z_src, z_dest = Z[0], Z[1]
	# Interpolate between source and destination
	step = (z_dest - z_src)/float(n_steps)
	results = []
	for i in range(n_steps+1):
		z = (z_src + step * i).reshape(1,-1) # Shape required by decoder
		x_hat = vae.decoder.predict(z)
		smi = one_hot_to_smiles(x_hat, index_to_char)[0] # One hot to smiles returns a list
		results.append(smi)
	return results


""" Remove duplicates from a list while preserving order
	"""
def remove_duplicates(l):
	seen = set()
	return [x for x in l if not (x in seen or seen.add(x))]



if __name__ == "__main__":


	""" PREPROCESS DATA 
		"""
	with open(TRAIN_FILE, 'r') as f:
		smiles = f.readlines()
	smiles = [i.strip() for i in smiles]	# Strip newlines
	if LIMIT is not None:               	# Select training set size
		smiles = smiles[:LIMIT]

	# Keep track of molecular descriptors corresponding to molecules
	logP = []
	SA = []
	for smi in smiles:
		mol = Chem.MolFromSmiles(smi)
		logP.append(Descriptors.MolLogP(mol))

	print('Training set size is {}'.format(len(smiles)))
	smiles = [filter_and_pad(i) for i in smiles if filter_and_pad(i)] # Filter large mols and pad sequences
	print('Training set size is {}, after filtering to max length of {}'.format(len(smiles), MAX_LEN))
	shuffle(smiles)							# Shuffle training set

	# Extract vocabulary : set of characters present in our molecules
	chars = sorted(list(set(''.join(smiles))))
	n_chars = 35 #len(chars)
	print('Number of chars: {}'.format(n_chars))

	# Dictionaries to convert chars to integers and vice versa
	char_to_index = {c: i for i, c in enumerate(chars)}
	index_to_char = {i: c for i, c in enumerate(chars)}

	# Populate our training set of sequences of one hot vectors
	X = np.zeros((len(smiles), MAX_LEN, n_chars), dtype=np.float32)
	for i, smile in enumerate(smiles):
		for t, char in enumerate(smile):
			X[i, t, char_to_index[char]] = 1


	""" DEFINE MODEL AND TRAIN 
		"""
	
	vae = SequentialVariationalAutoencoder2(
		n_time_steps=MAX_LEN, 
		n_features=n_chars,
		weights_file="save/weights.hdf5") # retrieve weights from hdf5 file

	if TRAIN:
		checkpoint = ModelCheckpoint(filepath='save/weights_.hdf5', # save weights to hdf5 file
									 verbose=1,
									 save_best_only=True)

		reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
	                                  factor = 0.5,
	                                  patience = 2,
	                                  min_lr = 0.00000005)

		vae.autoencoder.fit(
			X,
			X,
			validation_split=0.10,
			nb_epoch=NUM_EPOCHS,
			batch_size=BATCH_SIZE,
			callbacks=[checkpoint, reduce_lr])
	

	""" EVALUATE MODEL
		"""
	# Compute reconstruction accuracy
	print(compute_reconstruction_accuracy(vae, X, index_to_char, verbose=True))

	# Plot latent space projected to two dimensions
	plot_latent(vae, X, y=logP, verbose=True)

	# Interpolate between two molecules in latent space
	smi1, smi2 = one_hot_to_smiles(X[:2], index_to_char)
	print('Interpolating between two molecules in latent space')
	print('{} (source)'.format(smi1))
	print('{} (destination)'.format(smi2))
	interpolated = interpolate(X[:2], n_chars, char_to_index, index_to_char, vae)
	#print('Interpolated molecules')
	#for smi in interpolated:
	#	print(smi)
	print('{} interpolated molecules generated'.format(len(interpolated)))
	unique_interpolated = remove_duplicates(interpolated)
	print('{} unique'.format(len(unique_interpolated)))
	mols = [Chem.MolFromSmiles(smi) for smi in unique_interpolated if Chem.MolFromSmiles(smi)]
	print('{} decode to valid molecules'.format(len(mols)))
	image = Draw.MolsToGridImage(mols, molsPerRow=5)
	image.save('draw/interpolation.png', 'PNG')
	











