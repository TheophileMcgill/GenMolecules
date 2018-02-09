from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski, Descriptors, Crippen

# Suppress output from Chem.MolFromSmiles
RDLogger.DisableLog('rdApp.error')

def ro5(mol, alternate):
	""" Checks for compliance with each of Lipinski's rules of 5
	
		Parameters
		----------
		mol: rdkit.Chem.rdchem.Mol
			The molecule to check
		alternate: bool
			Whether to check the number of rotatable bonds and polar surface area,
			as opposed to simply the molecular weight. It has been suggested that 
			this is a more reliable criterion for drug-likeness
			
		Returns
		-------
		violations: dict
			A dictionary of values for the violated rules
				logp: Octanol-water partition coefficient	( >  5   )
				ndon: Number of hydrogen bond donors		( >  5   )
				nacc: Number of hydrogen bond acceptors		( >  10  )
				mass: Molecular weight (daltons)			( >= 500 )
				alternate:
					nrot: Number of rotatable bonds			( >  10  )
					tpsa: Polar surface area (angstroms)	( >  140 )
			If a measurement does not violate a rule, it is not included
	"""

	logp = Crippen.MolLogP(mol)
	mass = Descriptors.MolWt(mol)
	ndon = Lipinski.NumHDonors(mol)
	nacc = Lipinski.NumHAcceptors(mol)
	nrot = Lipinski.NumRotatableBonds(mol)
	tpsa = Descriptors.TPSA(mol)
	
	violations = {}
	if ndon > 5: violations['ndon'] = ndon
	if nacc > 10: violations['nacc'] = nacc
	if logp > 5: violations['logp'] = logp
	
	if alternate:
		if nrot > 10: violations['nrot'] = nrot
		if tpsa > 140: violations['tpsa'] = tpsa
	else:
		if mass >= 500: violations['mass'] = mass
	
	return violations

def ro3(mol):
	""" Checks for compliance with each of the "rules of 3" for lead-likeness
	
		Parameters
		----------
		mol: rdkit.Chem.rdchem.Mol
			The molecule to check
			
		Returns
		-------
		violations: dict
			A dictionary of values for the violated rules
				logp: Octanol-water partition coefficient	( >  3   )
				mass: Molecular weight (daltons)			( >= 300 )
				ndon: Number of hydrogen bond donors		( >  3   )
				nacc: Number of hydrogen bond acceptors		( >  3   )
				nrot: Number of rotatable bonds				( >  3   )
				tpsa: Polar surface area (angstroms)		( >  60  )
			If a measurement does not violate a rule, it is not included
	"""

	logp = Crippen.MolLogP(mol)
	mass = Descriptors.MolWt(mol)
	ndon = Lipinski.NumHDonors(mol)
	nacc = Lipinski.NumHAcceptors(mol)
	nrot = Lipinski.NumRotatableBonds(mol)
	tpsa = Descriptors.TPSA(mol)
	
	violations = {}
	if logp >= 3: violations['logp'] = logp
	if mass >= 300: violations['mass'] = mass
	if ndon > 3: violations['ndon'] = ndon
	if nacc > 3: violations['nacc'] = nacc
	if nrot > 3: violations['nrot'] = nrot
	if tpsa > 60: violations['tpsa'] = tpsa
	
	return violations

def druglike_smi(smi, alternate=False):
	""" Check for Rule of Five Compliance of SMILES string """
	mol = Chem.MolFromSmiles(smi)
	return mol is not None and not ro5(mol, alternate)

def leadlike_smi(smi):
	""" Check for Rule of Three Compliance of SMILES string """
	mol = Chem.MolFromSmiles(smi)
	return mol is not None and not ro3(mol)
