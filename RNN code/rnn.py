import numpy as np
np.random.seed(42)

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import adam

from generate import GenerationCallback, SmilesDataHandler, gen_with_model

# Molecule checking
from rdkit import Chem, RDLogger

# Logging
import sys, csv


if __name__ == "__main__":
	
	# Data handler
	dh = SmilesDataHandler(filename='data/chembl.smi',
						   timesteps=64, max_mol=128)
						   

	# Define the LSTM model
	model = Sequential()
	model.add(LSTM(1024, input_shape=dh.input_shape, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(1024, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(1024))
	model.add(Dropout(0.2))
	model.add(Dense(dh.F, activation='softmax'))
	
	model.compile(loss='categorical_crossentropy',
				  metrics=['accuracy'],
				  optimizer=adam(lr=0.001, clipnorm=5.))
	
	dh.save_chars('data/chars.txt')
	open('save/rnn/model.json','w').write(model.to_json())
	
	# Novel molecule validator
	RDLogger.DisableLog('rdApp.error')
	valid = lambda mol: Chem.MolFromSmiles(mol) is not None

	# Novel molecule generator
	gen = lambda m: gen_with_model(m, dh.chars, temp=0.65)
	
	# Using checkpoints, train the model
	callbacks_list = [
		ModelCheckpoint(
			filepath="save/rnn/weights-{epoch:02d}-{val_loss:.4f}.hdf5",
			monitor='val_loss',
			mode='min',
			save_best_only=True
		),
		ReduceLROnPlateau(
			monitor='val_loss',
			factor=0.5,
			patience=20,
			min_lr=0.000001
		),
		EarlyStopping(
			monitor='val_loss',
			min_delta=0,
			patience=50,
		),
		GenerationCallback(
			generator=gen,
			save_file='save/rnn/generated.csv',
			nb_batches=500,
			nb_samples=100,
			custom_funcs=[('Valid',valid)]
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
		writer.writerows(zip(*values))

