import os, sys
from smal.all import from_smi
import numpy as np
import pandas as pd
import torch
from dgl.convert import graph
from pathlib import Path
import pandas as pd
from rdkit import Chem
import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem, ChemicalFeatures
import numpy as np
import os
import torch


import numpy as np

import torch

from pathlib import Path
import joblib
import torch
import dgl
import numpy as np

                        
def collate_reaction_graphs(batch):
    g_mol, y = map(list, zip(*batch))
    g_mol = dgl.batch(g_mol)
    y = torch.FloatTensor(np.hstack(y))
    return g_mol, y


def MC_dropout(model):

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    pass



def store_result(obj:"any", experiment_name:str, fle_name:str) -> None:
    path_obj = path_for_experiment_and_fle(experiment_name=experiment_name,fle_name=fle_name,)
    joblib.dump(obj,path_obj)

def load_result(experiment_name:str, fle_name:str) -> "any":
    path_obj = path_for_experiment_and_fle(experiment_name=experiment_name,fle_name=fle_name,)
    return joblib.load(path_obj)




class SMDataset():

    def __init__(self, df:pd.DataFrame,):
        if len(df):
            self.is_empty_dataset = False
            self._read_dataframe(df)
        else:
            self.is_empty_dataset = True
            self.y = []
        


    def _read_dataframe(self, df:pd.DataFrame):
        def _read_mol_dict(molsuppl):
            def add_mol(mol_dict, mol):

                def _DA(mol):

                    D_list, A_list = [], []
                    for feat in chem_feature_factory.GetFeaturesForMol(mol):
                        if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
                        if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])
                    
                    return D_list, A_list

                def _chirality(atom):

                    if atom.HasProp('Chirality'):
                        #assert atom.GetProp('Chirality') in ['Tet_CW', 'Tet_CCW']
                        c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')] 
                    else:
                        c_list = [0, 0]

                    return c_list

                def _stereochemistry(bond):

                    if bond.HasProp('Stereochemistry'):
                        #assert bond.GetProp('Stereochemistry') in ['Bond_Cis', 'Bond_Trans']
                        s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')] 
                    else:
                        s_list = [0, 0]

                    return s_list    
                    

                n_node = mol.GetNumAtoms()
                n_edge = mol.GetNumBonds() * 2

                D_list, A_list = _DA(mol)
                rings = mol.GetRingInfo().AtomRings()
                atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
                atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
                atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
                atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
                atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors = True)) for a in mol.GetAtoms()]][:,:-1]
                atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
                atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
                atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
                atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
                atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
                
                node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

                #shift = np.array([atom.GetDoubleProp('shift') for atom in mol.GetAtoms()])
                #mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])

                mol_dict['n_node'].append(n_node)
                mol_dict['n_edge'].append(n_edge)
                mol_dict['node_attr'].append(node_attr)

                #mol_dict['shift'].append(shift)
                #mol_dict['mask'].append(mask)
                mol_dict['smi'].append(Chem.MolToSmiles(mol))
                
                if n_edge > 0:

                    bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
                    bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
                    bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]   
                    
                    edge_attr = np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1)
                    edge_attr = np.vstack([edge_attr, edge_attr])
                    
                    bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
                    src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
                    dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
                    
                    mol_dict['edge_attr'].append(edge_attr)
                    mol_dict['src'].append(src)
                    mol_dict['dst'].append(dst)
                
                return mol_dict


            atom_list = ['H', 'Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi']
            charge_list = [1, 2, 3, -1, -2, -3, 0]
            degree_list = [1, 2, 3, 4, 5, 6, 0]
            valence_list = [1, 2, 3, 4, 5, 6, 0]
            hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
            hydrogen_list = [1, 2, 3, 4, 0]
            ringsize_list = [3, 4, 5, 6, 7, 8]

            bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

            rdBase.DisableLog('rdApp.error') 
            rdBase.DisableLog('rdApp.warning')
            chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

            mol_dict = {'n_node': [],
                        'n_edge': [],
                        'node_attr': [],
                        'edge_attr': [],
                        'src': [],
                        'dst': [],
                        #'shift': [],
                        #'mask': [],
                        'smi': []}
                            
            for i, mol in enumerate(molsuppl):
                if isinstance(mol,str):
                    mol = from_smi(mol)

                try:
                    Chem.SanitizeMol(mol)
                    si = Chem.FindPotentialStereo(mol)
                    mol_probe = Chem.AddHs(mol)
                    atom_types = {atm.GetSymbol() for atm in mol_probe.GetAtoms()}
                    if 'H' not in atom_types and 'C' not in atom_types:
                        print("skipping smiles",Chem.MolToSmiles(mol_probe))
                        continue
                    for element in si:
                        if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                            mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
                        elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                            mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
                    #assert '.' not in Chem.MolToSmiles(mol) # TODO
                except:
                    raise

                mol = Chem.RemoveHs(mol)
                mol_dict = add_mol(mol_dict, mol)

                if (i+1) % 1000 == 0: print('%d/%d processed' %(i+1, len(molsuppl)))

            print('%d/%d processed' %(i+1, len(molsuppl)))   

            mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
            mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
            mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
            mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
            mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
            mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
            #mol_dict['shift'] = np.hstack(mol_dict['shift'])
            #mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
            mol_dict['smi'] = np.array(mol_dict['smi'])

            mol_dict['n_csum'] = np.concatenate([[0], np.cumsum(mol_dict['n_node'])])
            mol_dict['e_csum'] = np.concatenate([[0], np.cumsum(mol_dict['n_edge'])])

            for key in mol_dict.keys(): 
                print(key, mol_dict[key].shape, mol_dict[key].dtype)
            
            return mol_dict

        self.mol_dict = _read_mol_dict(df["smiles"])
        self.y = df["y"].values

    def _load_graph(self,mol_dict:dict,idx:int,):
        e_csum = mol_dict["e_csum"]
        n_csum = mol_dict["n_csum"]
        n_node = mol_dict["n_node"]
        src = mol_dict["src"]
        dst = mol_dict["dst"]
        node_attr = mol_dict["node_attr"]
        edge_attr = mol_dict["edge_attr"]
        g = graph((src[e_csum[idx]:e_csum[idx+1]], dst[e_csum[idx]:e_csum[idx+1]]), num_nodes = n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(node_attr[n_csum[idx]:n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(edge_attr[e_csum[idx]:e_csum[idx+1]]).float()
        return g

    def __getitem__(self, idx):
        y = self.y[idx]
        g_mol = self._load_graph(self.mol_dict,idx)
        return g_mol, y
        
        
    def __len__(self):
        return len(self.y)



import argparse
import joblib
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import SDWriter
import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset

from sklearn.metrics import mean_absolute_error,r2_score
from scipy import stats



from smal.all import random_fle
import numpy as np
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dgl.nn.pytorch import NNConv, Set2Set


from sklearn.metrics import mean_absolute_error,r2_score
from scipy import stats


import torch.nn as nn

from dgl.nn.pytorch import Set2Set

from dgllife.model.gnn.mpnn import MPNNGNN


class MPNNPredictor(nn.Module):
    """MPNN for regression and classification on graphs.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_hidden_feats : int
        Size for the hidden edge representations. Default to 128.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    num_step_set2set : int
        Number of set2set steps. Default to 6.
    num_layer_set2set : int
        Number of set2set layers. Default to 3.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6,
                 num_step_set2set=6,
                 num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        self.readout = Set2Set(input_dim=node_out_feats,
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.predict = nn.Sequential(
            nn.Linear(2 * node_out_feats, node_out_feats),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )



    #def forward(self, g, node_feats, edge_feats):
    def forward(self, g, ):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features.

        Returns
        -------
        float32 tensor of shape (G, n_tasks)
            Prediction for the graphs in the batch. G for the number of graphs.
        """
        node_feats = g.ndata['node_attr']
        edge_feats = g.edata['edge_attr']
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)


def training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path, n_forward_pass = 5, cuda = torch.device('cuda:0'), experiment_name=None):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    optimizer = Adam(net.parameters(), lr=1e-3,)# weight_decay=1e-10)

    max_epochs = 200
    val_y = np.hstack([inst[-1] for inst in iter(val_loader.dataset)])
    val_log = np.zeros(max_epochs)
    train_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        
        # training
        net.train()
        start_time = time.time()
        for batchidx, batchdata in enumerate(train_loader):

            optimizer.zero_grad()
            g_mol, y = batchdata
            
            y = (y - train_y_mean) / train_y_std
            
            g_mol = g_mol.to(cuda)
            y = y.to(cuda)
            
            predictions = net(g_mol) 
            
            loss = torch.abs(predictions.squeeze() - y).mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item() * train_y_std
            train_log[epoch] = train_loss


        #print('--- training epoch %d, processed %d/%d, loss %.3f, time elapsed(min) %.2f' %(epoch,  train_size, train_size, train_loss, (time.time()-start_time)/60))
    
        # validation
        val_y_pred = inference(net, val_loader, train_y_mean, train_y_std, n_forward_pass = n_forward_pass)
        val_loss = mean_absolute_error(val_y, val_y_pred)
        val_r2 = r2_score(val_y, val_y_pred)
        val_spearmanr = stats.spearmanr(val_y, val_y_pred)[0]
        
        val_log[epoch] = val_loss
        if epoch % 10 == 0: 
            print('--- validation epoch %d, processed %d, current MAE %.3f, current r2 %.3f, current spearr %.3f, best MAE %.3f, time elapsed(min) %.2f' %(epoch, val_loader.dataset.__len__(), val_loss, val_r2, val_spearmanr, np.min(val_log[:epoch + 1]), (time.time()-start_time)/60))
        
        #lr_scheduler.step(val_loss)
        
        # earlystopping
        if np.argmin(val_log[:epoch + 1]) == epoch:
            torch.save(net.state_dict(), model_path) 
        
        elif np.argmin(val_log[:epoch + 1]) <= epoch - 50:
            break

    print('training terminated at epoch %d' %epoch)
    net.load_state_dict(torch.load(model_path))

    from matplotlib import pyplot as plt
    plt.clf()
    plt.plot(val_log,"b-o")
    plt.plot(train_log,"r-o")
    plt.savefig(path_for_experiment_and_fle(experiment_name=experiment_name,fle_name="loss_curve.svg"))
    plt.clf()
    
    return net
    

def inference(net, test_loader, train_y_mean, train_y_std, n_forward_pass = 30, cuda = torch.device('cuda:0')):

    if not torch.cuda.is_available():
        cuda = torch.device("cpu")

    net.eval()
    #MC_dropout(net)
    with torch.no_grad():
        y_pred = []
        for batchidx, batchdata in enumerate(test_loader):

            g_mol, _ = batchdata
            g_mol = g_mol.to(cuda)
            
            predictions = net(g_mol) # n_nodes, y)
            y_pred.append(predictions.cpu().numpy())

    y_pred_inv_std = np.vstack(y_pred) * train_y_std + train_y_mean
    return y_pred_inv_std


def path_for_experiment_and_fle(experiment_name,fle_name):
    d = Path("storage") / experiment_name
    d.mkdir(exist_ok=True)
    return d / fle_name

from smal.all import add_split_by_col
def run_for_smiles(pred_smis:list[str], storage_path, experiment_name, use_pretrain = True, ):
    batch_size = 128
    here = Path(storage_path)
    here.mkdir(exist_ok=True,)
    (here / "checkpoints").mkdir(exist_ok=True,)
    model_path = str(here / 'checkpoints' / 'model.pt')
    model_metadata_path = str(here / 'checkpoints' / 'model_metadata.pkl')
    #if not os.path.exists(model_path): os.makedirs(model_path)

    data_pred = pd.DataFrame({"smiles": pred_smis, "y": [0 for _ in range(len(pred_smis))]})

    data_pred = SMDataset(data_pred)
    data = pd.read_csv("delaney-processed.csv")
    data["y"] = data['measured log solubility in mols per litre']

    print(data.columns)

    assert len(data)

    add_split_by_col(data,col="smiles",amount_train=0.7,amount_test=0.2,amount_val=0.1,random_seed=123,)

    df_test = data[data["split"] == "test"]
    df_train = data[data["split"] == "train"]
    df_val = data[data["split"] == "val"]

    train_set = SMDataset(df_train)
    val_set = SMDataset(df_val)
    test_set = SMDataset(df_test)
    data = SMDataset(data)

    assert len(train_set)
    assert len(val_set)
    assert len(test_set)
    
    #data = SMDataset(data)
    #train_set, val_set, test_set = split_dataset(data, data_split, shuffle=True, random_state=random_seed)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)
    pred_loader = DataLoader(dataset=data_pred, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    train_y = np.hstack([inst[-1] for inst in iter(train_loader.dataset)])
    train_y_mean = np.mean(train_y.reshape(-1))
    train_y_std = np.std(train_y.reshape(-1))

    node_dim = data.mol_dict["node_attr"].shape[1]
    edge_dim = data.mol_dict["edge_attr"].shape[1]
    net = MPNNPredictor(node_dim, edge_dim).cuda()

    print('-- CONFIGURATIONS')
    print('--- data_size:', data.__len__())
    print('--- train/val/test: %d/%d/%d' %(train_set.__len__(), val_set.__len__(), test_set.__len__()))
    print('--- use_pretrain:', use_pretrain)
    print('--- model_path:', model_path)

    joblib.dump({"train_y_mean": train_y_mean, "train_y_std": train_y_std}, model_metadata_path,)

    # training
    if not use_pretrain or not Path(model_path).exists():
        print('-- TRAINING')
        net = training(net, train_loader, val_loader, train_y_mean, train_y_std, model_path,experiment_name=experiment_name,)
    else:
        print('-- LOAD SAVED MODEL')
        net.load_state_dict(torch.load(model_path))

    # inference
    test_y = np.hstack([inst[-1] for inst in iter(test_loader.dataset)])
    test_y_pred = inference(net, test_loader, train_y_mean, train_y_std)
    test_mae = mean_absolute_error(test_y, test_y_pred)

    df_test["y_pred"] = test_y_pred

    # TODO: implement
    #evaluate_on_test(df_test,experiment_name=experiment_name,)

    print('-- RESULT')
    print('--- test MAE', test_mae)
    return inference(net,pred_loader, train_y_mean, train_y_std)



run_for_smiles(["CCCOCC","CCOCCCCCN",],"storage",experiment_name="test",use_pretrain=False,)
