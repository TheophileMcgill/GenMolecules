from rdkit import Chem
from rdkit.Chem import Draw
import cairo
import sys, math, os
from random import shuffle
from lipinski import *

def factors(n):
	return reduce(list.__add__,
		([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))

smiles_file = sys.argv[1]

mols = Chem.SmilesMolSupplier(smiles_file,titleLine=False)
mols = [m for m in mols]
mols = sorted(mols, key=lambda m: m.GetNumBonds(),reverse=True)
mols = sorted(mols, key=lambda m: int(druglike_smi(Chem.MolToSmiles(m))), reverse=True)
N = len(mols)
n,m = factors(N)[-2:]

img = Draw.MolsToGridImage(mols,molsPerRow=m,subImgSize=(200,200),legends=[str(i) for i in range(len(mols))])
img.show()


n = int(sys.argv[2]) if len(sys.argv)>2 else 7
N = n*n
to_remove = len(mols) - N
rejects = raw_input("Enter %d bad indices, space-separated\n  >  "%to_remove)

mols = [m for i,m in enumerate(mols) if str(i) not in rejects.split()]
mols = mols[:N]
shuffle(mols)

img = Draw.MolsToGridImage(mols,molsPerRow=n,subImgSize=(200,200))
img.save(os.path.splitext(smiles_file)[0]+'_%dx%d.png'%(n,n))
