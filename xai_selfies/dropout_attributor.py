import copy
import random
from rdkit import Chem

def mutations_dropout(mol,dropout=0.25,seed=None,max_iterations=50,):
    """
    
    >>> smi = "CCCO"
    >>> list(mutations_dropout(Chem.MolFromSmiles(smi)))
    [('C.O', [1, 2]), ('CCCO', []), ('C.CO', [1]), ('CC', [0, 3]), ('CO', [0, 1]), ('CC', [2, 3]), ('C', [0, 2, 3]), ('CCO', [0]), ('CCC', [3]), ('CC.O', [2]), ('C.C', [1, 3]), ('C.O', [0, 2])]

    """
    random.seed(seed)
    visited = []
    for _ in range(max_iterations):
        ais_deleted = []
        mut = copy.deepcopy(mol)
        for ai in range(mol.GetNumAtoms()):
            if random.random() < dropout:
                ais_deleted.append(ai)
                mut.GetAtomWithIdx(ai).SetAtomicNum(0)
        mut = Chem.DeleteSubstructs(mut, Chem.MolFromSmarts("[#0]"))
        mut = Chem.MolToSmiles(mut)
        if ais_deleted not in visited:
            visited.append(ais_deleted)
            yield mut,ais_deleted