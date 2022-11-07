import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Optional, Union, Dict

import torch
from torch_geometric.data import InMemoryDataset, download_url

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import (AllChem, Draw, MACCSkeys, rdmolops)

class MolecularDataset(InMemoryDataset):
    def __init__(self, root, filename, transform=None, pre_transform=None, pre_filter=None):
        self.filename = filename
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        ### TODO: How to deal with the number 128 (maybe in other tasks.)
        self.num_tasks = 128
        self.task_type = "binary classification"
        

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        # TODO: Need check later
        # self.data = pd.read_csv(self.raw_paths[0])
        return "processed_files.pt"

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])

        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        data_list = []

        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):   
    
            f = featurizer.featurize(row["smiles"])

            # Convert smiles in to PyG's Data object.
            # Some smiles cannot be featurized (such as [Cd+2] which has only one atom.)
            try:
                data = f[0].to_pyg_graph()
                data.edge_attr = data.edge_attr.to(torch.float32)
                data.edge_index = data.edge_index.to(torch.long)
                # Assign labels(torch.size([1, 128])) to data.y 
                data.y = torch.tensor(row[:128].to_numpy(dtype=np.float32), dtype=torch.float32).reshape(1, -1) # or torch.from_numpy
                # Adding additional features (node_feats, edge_feats) obtained from rdkit.Chem
                mol = Chem.MolFromSmiles(row["smiles"])
                
                # NOTE: create node_feats, edge_feats attribute to distinguish with x, edge_attr right?
                data.node_feats = self._get_node_features(mol) # TODO: None Handling
                data.edge_feats = self._get_edge_features(mol)
                data.new_edge_index = self._get_adjacency_info(mol)
                data_list.append(data)
            except Exception as e:
                print(e)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_node_features(self, mol):
        """
        Returns matrix of size (# nodes, # feature size per node)
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # get Atomic Number
            node_feats.append(atom.GetAtomicNum())
            # get Atomic Degree
            node_feats.append(atom.GetDegree())
            # get Formal Charge
            node_feats.append(atom.GetFormalCharge())
            # get Hybridization
            node_feats.append(atom.GetHybridization())
            # get Aromaticity
            node_feats.append(atom.GetIsAromatic())

            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        Returns matrix of size (#edges, # feature size per edge)
        """

        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # get Bond type
            edge_feats.append(bond.GetBondTypeAsDouble())
            # get Rings
            edge_feats.append(bond.IsInRing())

            all_edge_feats.append(edge_feats)

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self, mol):
        """
        Return adjacency matrix of mol (2, #edges * 2)
        """
        adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
        row, col = np.where(adj_matrix)
        coo = np.array(list(zip(row, col)))
        coo = np.reshape(coo, (2, -1))
        return torch.tensor(coo, dtype=torch.float)

    def get_idx_split(self, split: Optional[str] = None) -> Union[Dict[str, np.ndarray], np.ndarray]:
        ### TODO: How to fix the seed here
        ### TODO: Ratio!
        num_data = len(self)
        rng = np.random.default_rng()
        perm = rng.permutation(num_data)
        split_idx = dict()
        split_idx['train'] = perm[:int(0.8 * num_data)]
        split_idx['valid'] = perm[int(0.8 * num_data):int(0.9 * num_data)]
        split_idx['test'] = perm[int(0.9 * num_data):]
        return split_idx


if __name__ == "__main__":
    dataset = MolecularDataset(root="./custom_dataset", filename="mol.csv")
    print(dataset)
    print(len(dataset))
    data = dataset[0]
    print(dataset[0])
    split_idx = dataset.get_idx_split()
    print(dataset[split_idx["train"]])
    print("Type checking")
    print(data.x.dtype)
    print(data.y.dtype)
    print(data.edge_attr.dtype)
    print(data.edge_index.dtype)