from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit import DataStructs

import cPickle
import csv, sys

# Molecule checking
from rdkit import Chem, RDLogger


RDLogger.DisableLog('rdApp.error')

with open('data/chembl_22_1_chemreps.txt','r') as f, open('data/chembl_full.smi','w') as g:
	# Discard header
	next(f)
	
	# Load in SMILES
	lines = (l.split()[:2] for l in f)
	
	# Convert to RDKit Molecules
	mols = ((Chem.MolFromSmiles(s),n) for n,s in lines)
	
	# Convert back into SMILES
	smiles = ((Chem.MolToSmiles(m),n) for m,n in mols if m)
	
	# Save to file
	lines = ("%s %s\n"%(s,n) for s,n in smiles)
	g.writelines(lines)



