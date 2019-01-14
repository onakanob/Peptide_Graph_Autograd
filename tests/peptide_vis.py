import rdkit.Chem as chem
from rdkit.Chem.Draw import SimilarityMaps, MolToMPL

# A_smiles = 'C(CC(C(=O)O)N)CN=C(N)N'
# molSmiles = chem.MolFromSmiles(A_smiles)

molAmino = chem.MolFromSequence('MPT')
print(molAmino.GetNumAtoms())

# contribs = chem.rdMolDescriptors._CalcCrippenContribs(molSmiles)
# fig = SimilarityMaps.GetSimilarityMapFromWeights(molSmiles,[x for x,y in contribs], colorMap='jet', contourLines=10)

#contribs = chem.rdMolDescriptors._CalcCrippenContribs(molAmino)
#fig = SimilarityMaps.GetSimilarityMapFromWeights(
#    molAmino, [x for x, y in contribs], colorMap='jet', contourLines=10)

fig = MolToMPL(molAmino, fitImage=True)
fig.savefig('testfig.png', bbox_inches='tight')
print('fig saved')
