from chembl_webresource_client.new_client import new_client


#accepts chembl id
# returns dict of chembl ID : smiles
def get_smiles(chembl_id):	
	molecule = new_client.molecule
	# print (molecule)
	m = molecule.get(chembl_id)
	smiles = m['molecule_structures']['canonical_smiles']
	return smiles


	
