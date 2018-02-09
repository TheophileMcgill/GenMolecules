import numpy as np
from keras.utils import np_utils
from keras.callbacks import Callback

from random import shuffle
import sys, csv
import json



class SmilesDataHandler(object):
	def __init__(self, filename, timesteps=100, max_mol=np.infty, chars=None,
				 hold_out=0.2):
	
		# Read from file
		with open(filename,'r') as f:
			lines = (l.strip() for l in f)					# Strip newlines
			lines = (l.split()[0] for l in lines)			# Remove name column
			lines = [l for l in lines if len(l) < max_mol]	# Filter large mols
			lines = list(set(lines))						# Remove duplicates
		
		# Shuffle the lines
		shuffle(lines)
			
		# Filter out OOV
		if chars is not None:
			cs = set(chars)
			lines = [l for l in lines if cs.issuperset(l)]
		
		# Get integer encoding of unique characters
		if chars is None:
			chars = np.unique([c for l in lines for c in l]+['\n'])
		self.chars = chars
		trans = np.vectorize(self.chars.tolist().index)
		
		# Hold out test set
		self.n_held_out = int(hold_out*len(lines))
		self.test_set = lines[:self.n_held_out]
		self.train_set = lines[self.n_held_out:]
		
		# Create raw text
		raw = '\n'.join(self.train_set)+'\n'
		self.stream = trans(list(raw))
		
		# Append the first T characters, to simulate circular indexing
		self.stream = np.append(self.stream, self.stream[:timesteps])
		
		# Dimensions of the data
		#	F = Number of features (unique characters)
		#	T = Number of timesteps unrolled
		#	B = Number of samples per batch
		#	N = Number of chatacters in the stream
		self.F = len(self.chars)
		self.T = timesteps
		self.N = len(self.stream)
		
		self.input_shape = (self.T,self.F)
		
	def save_chars(self, filename):
		open(filename,'w').write(''.join(self.chars.tolist()))
	
	@staticmethod
	def load_chars(filename):
		return np.array([c for l in open(filename,'r') for c in l])
	
	def batch_generator(self, batch_size=100):
		""" Generate batches of time series input output pairs
				
			Yields
			------
			X: Numpy array
				The input sequences in one-hot vector format
				Shape of X equal to (batch_size, timesteps, nb_features)
			y: Numpy array
				The output characters in one-hot vector format
				Shape of y equal to (batch_size, nb_features)
				
			Example
			-------
			stream = 'hello world', timesteps=5, batch_size=3
				First the string is unrolled
					   X	|  y
					--------+------
					'hello' | ' '
					'ello ' | 'w'
					'llo w' | 'o'
				Then each character is converted to one-hot vector format. The 
				third sample above ('llo w','o') would be converted into
					X[3] = [[0, 0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1]]
					y[3] = [0, 0, 0, 0, 0, 1, 0, 0]
				The last dimension of these arrays are the number of unique 
				characters in the original string. Using the stored chars array,
				an encoding can be restored to its original format
					chars[X.argmax(2)] = [['h', 'e', 'l', 'l', 'o'],
										  ['e', 'l', 'l', 'o', ' '],
										  ['l', 'l', 'o', ' ', 'w']]
					chars[y.argmax(1)] = [' ', 'w', 'o']
		"""
		
		while True:
			# Sample B random start positions without replacement
			st = np.random.choice(self.N, size=batch_size, replace=False)
			
			# Get the indices, unrolled over T timesteps
			X_idx = (np.arange(self.T) + st[:,None]) % self.N
			y_idx = (st + self.T) % self.N
			
			# Get the characters from the stream
			X = self.stream[X_idx]
			y = self.stream[y_idx]
				
			# One-hot vectorize
			y = np_utils.to_categorical(y, nb_classes=self.F)
			X = np_utils.to_categorical(X, nb_classes=self.F)
			X = np.reshape(X, (batch_size, self.T, self.F))
				
			# Generate
			yield (X,y)

def gen_with_model(model, chars, seed=None, temp=1.0):
	"""
	
		Parameters
		----------
		model: keras model
			The model to generate with.
		temp: float (0,1]
			Temperature for stochastic generation. Temperatures close to 0
			lead to more deterministic generation, while numbers close to 
			1 generate more randomly. The default is 1, which leads the 
			generator to choose characters with probability equal to the
			output of the net (assuming softmax output activation).
			When temp is zero, will always generate the argmax.
	"""
	if temp <= 0:
		raise ValueError("Temperature must be positive")
	
	T,F = model.input_shape[-2:]

	# Helper generator
	def gen_chars(x):
		while True:
			probs = model.predict(x).flatten()
			
			if temp:
				# Scale the log probs by temp to tune randomness
				probs = np.exp(np.log(probs) / temp)
				probs = probs / probs.sum()
				i = np.random.choice(F, p=probs)
			else:
				i = np.argmax(probs)
				
			# Add to input
			probs[:] = 0
			probs[i] = 1
			x[:] = np.hstack((x[:,1:], probs[None,None]))

			yield chars[i]

	mol = seed
	
	# If no seed provided, seed with all newlines
	if seed is None:
		mol = ''
		seed = '\n'*T
	
	# Translate seed to integer encoding
	trans = np.vectorize(chars.tolist().index)
	seed = trans(list(seed))
	
	# One-hot encode it
	seed = np_utils.to_categorical(seed, nb_classes=F)[None,:,:]
	
	# If seed longer than necessary, just keep the tail
	if seed.shape[1] > T:
		seed = seed[:,-T:,:]
	
	# If seed to short, prefix with random padding
	if seed.shape[1] < T:
		n_pad = T - seed.shape[1]
		padding = np.random.random((1,n_pad,F))
		padding = padding / padding.sum(axis=-1, keepdims=True)
		seed = np.hstack((padding,seed))
		
	gen = gen_chars(seed)

	# Generate forever
	while True:
		c = next(gen)
		if c == '\n':
			if mol:
				yield mol
			mol = ''
		else:
			mol += c


class GenerationCallback(Callback):
	def __init__(self, generator, save_file, nb_batches=1000, nb_samples=1,
				 custom_funcs=[]):
		"""
		
			Parameters
			----------
			generator: function: model -> generator
				A function that creates a python generator when given a model
			save_file: str
				A .csv filename to save the results
			nb_batches: int
				How often to generate. By default, generates every 1000 batches
			nb_samples: int
				Number of samples to generate
			custom_funcs: [(str, function: str -> ?)]
				List of (string,function) tuples. Each function should take a
				string of the form of the generated output and return a value
				that should be logged to the CSV file. The string should 
				describe the function, and will be the column header
			
		"""
		super(GenerationCallback, self).__init__()

		self.generator = generator
		self.save_file = save_file
		self.nb_samples = nb_samples
		self.nb_batches = nb_batches
		func_names, self.custom_funcs = zip(*custom_funcs)
		
		self.batch_count = 0
	
		with open(self.save_file,'w') as f:
			writer = csv.writer(f)
			writer.writerow(['Batch','Generated Example']+list(func_names))

	def on_train_begin(self, logs=None):
		gen = self.generator(self.model)
		with open(self.save_file, 'a') as f:
			writer = csv.writer(f)
			for i in range(self.nb_samples):
				g = next(gen)
				customs = [f(g) for f in self.custom_funcs]
				writer.writerow([self.batch_count,g]+customs)
	
	def on_batch_end(self, batch, logs=None):
		self.batch_count += 1
		if self.batch_count % self.nb_batches == 0:
			gen = self.generator(self.model)
			with open(self.save_file,'a') as f:
				writer = csv.writer(f)
				for i in range(self.nb_samples):
					g = next(gen)
					customs = [f(g) for f in self.custom_funcs]
					writer.writerow([self.batch_count,g]+customs)


if __name__=="__main__":
	np.random.seed(42)

	import sys, getopt
	from keras.models import model_from_json
	from lipinski import druglike_smi, leadlike_smi
	
	# Molecule checking
	from rdkit import Chem, RDLogger
	
	args = sys.argv[1:]
	opts, args = getopt.getopt(args,"hc:m:w:n:o:t:",["chars=","model=","weights=","ngen=","output=","temp="])
	
	mfile = None
	wfile = None
	cfile = None
	ofile = None
	ngen = 10
	temp = 0.65
	for opt, arg in opts:
		if opt == '-h':
			print "Usage:"
			print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5> -o <output.smi> [-n <# samples>] [-t <temp>]"
			sys.exit()
		elif opt in ("-c", "--chars"):
			cfile = arg
		elif opt in ("-m", "--model"):
			mfile = arg
		elif opt in ("-w", "--weights"):
			wfile = arg
		elif opt in ("-o", "--output"):
			ofile = arg
		elif opt in ("-n", "--ngen"):
			ngen = int(arg)
		elif opt in ("-t", "--temp"):
			temp = float(arg)

	if cfile is None:
		print "Must provide character file!"
		print "Usage:"
		print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5> -o <output.smi> [-n <# samples>] [-t <temp>]"
		sys.exit()

	if mfile is None:
		print "Must provide a model!"
		print "Usage:"
		print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5> -o <output.smi> [-n <# samples>] [-t <temp>]"
		sys.exit()

	if wfile is None:
		print "Must provide weights!"
		print "Usage:"
		print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5> -o <output.smi> [-n <# samples>] [-t <temp>]"
		sys.exit()

	if ofile is None:
		print "Must provide output file!"
		print "Usage:"
		print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5> -o <output.smi> [-n <# samples>] [-t <temp>]"
		sys.exit()

	chars = SmilesDataHandler.load_chars(cfile)
	with open(mfile, 'r') as f:
		model_json = f.read()
		model = model_from_json(model_json)
	model.load_weights(wfile)

	model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

	# Novel molecule validator
	RDLogger.DisableLog('rdApp.error')
	valid = lambda mol: Chem.MolFromSmiles(mol) is not None

	# Generate molecules
	gen  = gen_with_model(model, chars, temp=temp)
	valids,druglike,leadlike = 0.0, 0.0, 0.0
	with open(ofile,'w') as f:
		for i in range(1,ngen+1):
			smi = next(gen)
			
			v = valid(smi)
			d = druglike_smi(smi)
			l = leadlike_smi(smi)
			
			valids += float(v)
			druglike += float(d)
			leadlike += float(l)
			
			sys.stdout.write("\r{:d} Generated: {:.2%} valid, {:.2%} druglike, {:.2%} leadlike ".format(i,valids/i,druglike/i,leadlike/i))
			sys.stdout.flush()
			
			if v:
				f.write('%s mol%d\n'%(smi,i))
				f.flush()
	print ""




