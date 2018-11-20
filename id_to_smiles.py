import h5py
from scipy import sparse
import query_chembl


kinases = ['cdk2', 'egfr_erbB1', 'gsk3b', 'hgfr', 'map_k_p38a', 'tpk_lck', 'tpk_src', 'vegfr2']
for kinase in kinases:
	print(kinase)
	hf = h5py.File("data/%s.h5" % (kinase), "r")
	ids = list(hf["chembl_id"].value) # the name of each molecules
	labels = hf["label"].value
	ids = [i.decode('UTF-8') for i in ids]
	with open('data/%s_smiles.csv' % (kinase), 'w') as f:
		for i in range(len(ids)):
			try:
				smiles = query_chembl.get_smiles(ids[i])
				f.write('%s,%s,%s\n' % (ids[i], smiles, labels[i]))
				print(i, ids[i],smiles)
			except: 
				print(i, ids[i], '******ERROR******')

