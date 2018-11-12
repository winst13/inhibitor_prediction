from chembl_webresource_client.new_client import new_client


#accepts list of numpy float list of chembl accession
# returns dict of chembl ID : smiles
def get_smiles(chembl_id_list):
	chembl_id_list = [i.decode('UTF-8') for i in chembl_id_list]
	molecule = new_client.molecule
	# print (molecule)
	molecules = molecule.get(chembl_id_list)
	id_to_smiles = {m['molecule_chembl_id']: m['molecule_structures']['canonical_smiles'] for m in molecules}
	return id_to_smiles


	
