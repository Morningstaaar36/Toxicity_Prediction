import numpy as np
import torch
from rdkit import Chem
from scipy import sparse as sp
from torch_geometric.data import Data


class SMILESToPyG:


    def __init__(self, pos_enc_dim=8, use_chirality=True, explicit_H=False):
        self.pos_enc_dim = pos_enc_dim
        self.use_chirality = use_chirality
        self.explicit_H = explicit_H

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return [x == s for s in allowable_set]

    def one_of_k_encoding_unk(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]

    def atom_features(self, atom):

        symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
        degree = [0, 1, 2, 3, 4, 5, 6]
        hybridizationType = [Chem.rdchem.HybridizationType.SP,
                             Chem.rdchem.HybridizationType.SP2,
                             Chem.rdchem.HybridizationType.SP3,
                             Chem.rdchem.HybridizationType.SP3D,
                             Chem.rdchem.HybridizationType.SP3D2,
                             'other']

        results = self.one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
                  self.one_of_k_encoding(atom.GetDegree(), degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  self.one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                      atom.GetIsAromatic()]

        if not self.explicit_H:
            results = results + self.one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                          [0, 1, 2, 3, 4])
        if self.use_chirality:
            try:
                results = results + self.one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

        return results

    def bond_features(self, bond):
        bt = bond.GetBondType()
        bond_feats = [
            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]
        if self.use_chirality:
            bond_feats = bond_feats + self.one_of_k_encoding_unk(
                str(bond.GetStereo()),
                ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        return np.array(bond_feats).astype(int)

    def laplacian_positional_encoding(self, edge_index, num_nodes):

        edge_index_np = edge_index.numpy()
        A = sp.csr_matrix(
            (np.ones(edge_index_np.shape[1]),
             (edge_index_np[0], edge_index_np[1])),
            shape=(num_nodes, num_nodes)
        ).astype(float)


        degrees = np.array(A.sum(axis=0)).flatten()

        N = sp.diags(degrees.clip(1) ** -0.5, dtype=float)


        L = sp.eye(num_nodes) - N * A * N


        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort()
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])


        if EigVec.shape[1] < self.pos_enc_dim + 1:
            PadVec = np.zeros((EigVec.shape[0], self.pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
            EigVec = np.concatenate((EigVec, PadVec), 1)

        lap_pos_enc = torch.from_numpy(EigVec[:, 1:self.pos_enc_dim + 1]).float()

        return lap_pos_enc

    def convert(self, smiles):

        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")

        num_atoms = mol.GetNumAtoms()

        atom_feats = np.array([self.atom_features(a) for a in mol.GetAtoms()])

        if self.use_chirality:
            chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                      useLegacyImplementation=False)
            chiral_arr = np.zeros([num_atoms, 3])
            for (i, rs) in chiralcenters:
                if rs == 'R':
                    chiral_arr[i, 0] = 1
                elif rs == 'S':
                    chiral_arr[i, 1] = 1
                else:
                    chiral_arr[i, 2] = 1
            atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

        x = torch.tensor(atom_feats, dtype=torch.float32)

        src_list = []
        dst_list = []
        bond_feats_all = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_feats = self.bond_features(bond)
            src_list.extend([u, v])
            dst_list.extend([v, u])
            bond_feats_all.append(bond_feats)
            bond_feats_all.append(bond_feats)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.array(bond_feats_all), dtype=torch.float32)

        laplacian_pe = self.laplacian_positional_encoding(edge_index, num_atoms)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            laplacian_pe=laplacian_pe,
            smiles=smiles
        )

        return data
