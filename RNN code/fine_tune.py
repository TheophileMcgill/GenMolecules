import numpy as np
np.random.seed(42)

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import adam
from keras.models import model_from_json
from rdkit import Chem, RDLogger
import sys, csv

from generate import SmilesDataHandler, GenerationCallback, gen_with_model
from lipinski import druglike_smi, leadlike_smi

# Logging
import sys, csv

if __name__ == "__main__":
	
	# Load pretrained model
	with open('save/rnn/model.json', 'r') as f:
		model_json = f.read()
		model = model_from_json(model_json)
	model.load_weights("save/rnn/weights_pre-finetuning.hdf5")
	model.compile(loss='categorical_crossentropy',
				  metrics=['accuracy'],
				  optimizer=adam(lr=0.001, clipnorm=5.))
	T,F = model.input_shape[-2:]

	# Data handler
	chars = SmilesDataHandler.load_chars('data/chars.txt')
	dh = SmilesDataHandler(filename='data/leadlike.smi',
						   timesteps=T, chars=chars)
		
	# Novel molecule generator
	gen = lambda m: gen_with_model(m, chars, temp=0.65)
	
	# Check if in held-out test set
	test_mols = map(Chem.MolFromSmiles,dh.test_set)
	train_mols = map(Chem.MolFromSmiles,dh.train_set)
	in_test_set = lambda smi: Chem.MolFromSmiles(smi) in test_mols
	in_train_set = lambda smi: Chem.MolFromSmiles(smi) in train_mols
	
	# Using checkpoints, train the model
	callbacks_list = [
		ModelCheckpoint(
			filepath="save/rnn/leadlike-weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5",
			monitor='val_loss',
			mode='min',
			save_best_only=True
		),
		ReduceLROnPlateau(
			monitor='val_loss',
			factor=0.5,
			patience=10,
			min_lr=0.000001
		),
		EarlyStopping(
			monitor='val_loss',
			min_delta=0,
			patience=20,
		),
		GenerationCallback(
			generator=gen,
			save_file='save/rnn/leadlike-generated.csv',
			nb_batches=500,
			nb_samples=100,
			custom_funcs=[('Druglike',druglike_smi),('Leadlike',leadlike_smi),("In Train",in_train_set),("In Test",in_test_set)]
		)
	]
	hist = model.fit_generator(
		generator=dh.batch_generator(batch_size=128),
		samples_per_epoch=8192,
		validation_data=dh.batch_generator(batch_size=128),
		nb_val_samples=1024,
		nb_epoch=2048,
		callbacks=callbacks_list,
		verbose=1
	)
	
	fields,values = zip(*hist.history.iteritems())
	
	with open('save/rnn/history.csv','w') as f:
		writer = csv.writer(f)
		writer.writerow(fields)
