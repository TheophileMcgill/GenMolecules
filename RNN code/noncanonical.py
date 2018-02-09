from rdkit import Chem

# SMARTS for rotatable bonds
rot_patt = Chem.MolFromSmarts('[!$(*#*)&!D1]-!@[!$(*#*)&!D1]')

with open('data/chembl.smi','r') as f, open('data/noncanonical.smi','w') as g:
	# Load in SMILES
	smiles = (l.strip().split()[0] for l in f)
	
	# Convert to RDKit Molecules
	mols = (Chem.MolFromSmiles(m) for m in smiles)
	
	for i in range(50):
		mol = next(mols)
		forms = set()
		print i
		for j in range(mol.GetNumAtoms()):
			try:
				forms.add(Chem.MolToSmiles(mol,rootedAtAtom=j))
			except:
				break
		for f in forms:
			g.write('%s mol%d\n'%(f,i))
