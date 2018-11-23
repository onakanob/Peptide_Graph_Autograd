import rdkit.Chem as chem
from rdkit.Chem.Draw import SimilarityMaps, MolToMPL

A_smiles = 'C(CC(C(=O)O)N)CN=C(N)N'

molSmiles = chem.MolFromSmiles(A_smiles)
molAmino = chem.MolFromSequence('VLQRNCAAYL')
molAmino2 = chem.MolFromFASTA('R')

print molSmiles is None
print molAmino is None
print molAmino2 is None

print molSmiles.GetNumAtoms()
print molAmino.GetNumAtoms()
print molAmino2.GetNumAtoms()

contribs = chem.rdMolDescriptors._CalcCrippenContribs(molSmiles)
fig = SimilarityMaps.GetSimilarityMapFromWeights(molSmiles,[x for x,y in contribs], colorMap='jet', contourLines=10)

contribs = chem.rdMolDescriptors._CalcCrippenContribs(molAmino)
fig = SimilarityMaps.GetSimilarityMapFromWeights(molAmino,[x for x,y in contribs], colorMap='jet', contourLines=10)

contribs = chem.rdMolDescriptors._CalcCrippenContribs(molAmino2)
fig = SimilarityMaps.GetSimilarityMapFromWeights(molAmino2,[x for x,y in contribs], colorMap='jet', contourLines=10)

MolToMPL(molAmino, size=(300, 300))