import numpy as np
from keras.utils import np_utils
from keras.callbacks import Callback

from random import shuffle
import sys, csv
import json

from generate import gen_with_model, SmilesDataHandler

if __name__=="__main__":
	np.random.seed(42)

	import sys, getopt
	from keras.models import model_from_json
	from lipinski import druglike_smi, leadlike_smi
	
	# Molecule checking
	from rdkit import Chem, RDLogger
	
	args = sys.argv[1:]
	opts, args = getopt.getopt(args,"hc:m:w:",["chars=","model=","weights="])
	
	mfile = None
	wfile = None
	cfile = None
	for opt, arg in opts:
		if opt == '-h':
			print "Usage:"
			print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5>"
			sys.exit()
		elif opt in ("-c", "--chars"):
			cfile = arg
		elif opt in ("-m", "--model"):
			mfile = arg
		elif opt in ("-w", "--weights"):
			wfile = arg

	if cfile is None:
		print "Must provide character file!"
		print "Usage:"
		print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5>"
		sys.exit()

	if mfile is None:
		print "Must provide a model!"
		print "Usage:"
		print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5>"
		sys.exit()

	if wfile is None:
		print "Must provide weights!"
		print "Usage:"
		print "   generate.py -c <chars.txt> -m <model.json> -w <weights.hdf5>"
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
	valids,druglike,leadlike = 0.0, 0.0, 0.0
	with open('save/rnn/leadlike_temperature.csv','w') as f:
		writer = csv.writer(f)
		writer.writerow(["Temperature","Valid","Druglike","Leadlike","Length"])
		for t in np.arange(1.0,0.34,-0.025):
			gen  = gen_with_model(model, chars, temp=t)
			next(gen) # Get rid of the first one, usually garbage
			n_valid = 0.0
			n_druglike = 0.0
			n_leadlike = 0.0
			total_length = 0.0
			for i in range(1,101):
				smi = next(gen)
				
				n_valid += valid(smi)
				n_druglike += druglike_smi(smi)
				n_leadlike += leadlike_smi(smi)
				total_length += len(smi)
				
				sys.stdout.write("\rT = {:.3f}: {:d} Generated: {:.2%} valid, {:.2%} druglike, {:.2%} leadlike, {:.2f} average length   ".format(
					t, i,
					n_valid/i,
					n_druglike/i,
					n_leadlike/i,
					total_length/i
				))
				sys.stdout.flush()
			print ""
			writer.writerow([
				t,
				n_valid/100.0,
				n_druglike/100.0,
				n_leadlike/100.0,
				total_length/100.0
			])




