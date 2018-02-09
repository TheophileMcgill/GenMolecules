from rdkit import Chem
from lipinski import druglike_smi, leadlike_smi
import sys

with open('data/chembl_full.smi','r') as f, open('data/druglike.smi','w') as g:
	# Discard header
	next(f)

	# Load in SMILES
	smiles = (l.strip().split() for l in f)
	
	# Get only drug-like molecules
	def get_druglike(smiles):
		n_good = 0.0
		total = 0
		for s,n in smiles:
			good = druglike_smi(s)
			n_good += good
			total += 1
			sys.stdout.write('\r{:d} checked, {:.2%} accepted'.format(total,n_good/total))
			sys.stdout.flush()
			if good: yield (s,n)
	
	# Save to file
	lines = ("%s %s\n"%(s,n) for s,n in get_druglike(smiles))
	g.writelines(lines)

print ""
