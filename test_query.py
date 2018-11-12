import h5py
from scipy import sparse
import query_chembl

hf = h5py.File("cdk2.h5", "r")
ids = list(hf["chembl_id"].value) # the name of each molecules
smiles = query_chembl.get_smiles(ids[:3])
print(smiles)