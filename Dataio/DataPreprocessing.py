import os 
import math
import Bio.PDB
import pickle as pkl
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from Bio.PDB import Selection
from tools import dictionary_covalent_bonds_numba
from numba.typed import List,Dict
from numba import types

list_aa = [
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y']

aa_to_index = dict([(list_aa[i],i) for i in range(20)])

residue_dictionary = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
                      'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                      'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
hetresidue_field = [' '] + ['H_%s'%name for name in residue_dictionary.keys()]

list_atoms_types = ['C', 'O', 'N', 'S']
atom_mass = np.array([12, 16, 14, 32])
atom_type_to_index = dict([(list_atoms_types[i], i)
                           for i in range(len(list_atoms_types))])
list_atoms = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
              'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2',
              'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1',
              'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SE', 'SG']

atom_to_index = dict([(list_atoms[i], i) for i in range(len(list_atoms))])
atom_to_index['OT1'] = atom_to_index['O']
atom_to_index['OT2'] = atom_to_index['OXT']

index_to_type = np.zeros(38, dtype=int)
for atom,index in atom_to_index.items():
    index_to_type[index] = list_atoms_types.index(atom[0])

atom_type_mass = np.zeros( 38 )
for atom,index in atom_to_index.items():
    atom_type_mass[index] = atom_mass[index_to_type[index]]

def is_residue(residue):
    try:
        return (residue.get_id()[0] in hetresidue_field) & (residue.resname in residue_dictionary.keys())
    except:
        return False
    
def is_heavy_atom(atom):
    # Second condition for Rosetta-generated files.
    try:
        return (atom.get_id() in atom_to_index.keys() )
    except:
        return False
    
def is_hydrogen(atom):
    atomid = atom.get_id()
    if len(atomid)>0:
        cond1 = (atomid[0] == 'H')
        cond2a = (atomid[0] in ['*','0','1','2','3','4','5','6','7','8','9'])
    else:
        cond1 = False
        cond2a = False
    if len(atomid)>1:
        cond2b = atomid[1] == 'H'
    else:
        cond2b = False
    return cond1 | (cond2a & cond2b)

def load_chains(pdb_id=None,
                         chain_ids='all',
                         file=None,
                         pdbparser=None, mmcifparser=None,
                         dockground_indexing=False, biounit=True,verbose=True):
    if pdbparser is None:
        pdbparser = Bio.PDB.PDBParser()  # PDB parser; to read pdb files.
    if mmcifparser is None:
        mmcifparser = Bio.PDB.MMCIFParser()

    assert (file is not None) | (pdb_id is not None)

    if (file is None) & (pdb_id is not None):
        file = getPDB(pdb_id, biounit=biounit, structures_folder=structures_folder)[0]
    else:
        pdb_id = file.split('/')[-1].split('.')[0][-4:]

    if file[-4:] == '.cif':
        parser = mmcifparser
    else:
        parser = pdbparser
    with warnings.catch_warnings(record=True) as w:
        structure = parser.get_structure(pdb_id,  file)

    chain_objs = []
    if chain_ids in ['all','lower','upper']:
        for model_obj in structure:
            for chain_obj in model_obj:
                condition1 = (chain_ids == 'all')
                condition2 = ( (chain_ids == 'lower') & chain_obj.id.islower() )
                condition3 = ( (chain_ids == 'upper') & (chain_obj.id.isupper() | (chain_obj.id == ' ') ) )
                if condition1 | condition2 | condition3:
                    chain_objs.append(chain_obj)
    else:
        for model, chain in chain_ids:
            if dockground_indexing & (model > 0):
                model_ = model - 1
            else:
                model_ = model

            chain_obj = structure[model_][chain]
            chain_objs.append(chain_obj)
    return chain_objs


def process_chain(chain):
    backbone_coordinates = []
    all_coordinates = []
    all_atoms = []
    all_atom_types = []
    for residue in Selection.unfold_entities(chain, 'R'):
        if is_residue(residue):
            residue_atom_coordinates = np.array(
                [atom.get_coord() for atom in residue if is_heavy_atom(atom)])
            residue_atoms = [atom_to_index[atom.get_id()]
                             for atom in residue if is_heavy_atom(atom)  ]
            all_coordinates.append(residue_atom_coordinates)
            all_atoms.append(residue_atoms)
    return all_coordinates, all_atoms 


def get_aa_frame(atom_coordinates, atom_ids, pdb):
    aa_C, aa_PC, aa_SCoM = get_aa_frame_backbone(List(atom_coordinates), List(atom_ids), pdb)
    aa_C = torch.Tensor(aa_C)
    aa_PC = torch.Tensor(aa_PC)
    aa_SCoM = torch.Tensor(aa_SCoM)
    if len(aa_C) < 2 :
        return aa_C, _
    fm_z = aa_SCoM - aa_C 
    z_axis = fm_z / torch.norm(fm_z, p=2, dim=1).unsqueeze(-1)
    fm_y = torch.cross(z_axis, aa_PC-aa_SCoM)
    y_axis = fm_y / torch.norm(fm_y, p=2, dim=1).unsqueeze(-1)
    fm_x = torch.cross(y_axis, z_axis)
    x_axis = fm_x / torch.norm(fm_x, p=2, dim=1).unsqueeze(-1)
    local_coordinate = torch.stack([x_axis, y_axis, z_axis], dim=2)
    return aa_C, local_coordinate

def get_aa_frame_backbone(atom_coordinates, atom_ids, pdb):
    #Skeleton atomic coordinates
    L = len(atom_coordinates)
    aa_C = List()
    aa_PC = List()
    aa_SCoM = List()

    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        if l > 0:
            patom_coordinate = atom_coordinates[l-1]
            patom_id = atom_ids[l-1]
            PC_coodinate = patom_coordinate[patom_id.index(1)]
        else:
            PC_coodinate = atom_coordinates[0][1] + (atom_coordinates[1][1] - atom_coordinates[2][1]) 
        natoms = len(atom_id)
        if all(item in atom_id for item in [0, 1, 17]):
            C_coordinate = atom_coordinate[atom_id.index(0)]
            Calpha_coordinate = atom_coordinate[atom_id.index(1)]
            N_coordinate = atom_coordinate[atom_id.index(17)]
        else:
            print('Warning, pathological amino acid missing Calpha or N or C', l)
            print('have mistake:',pdb)
            break
            
        x_SCoM = 3*Calpha_coordinate - C_coordinate - N_coordinate
        aa_C.append(Calpha_coordinate)
        aa_PC.append(PC_coodinate)
        aa_SCoM.append(x_SCoM)
        
    return aa_C, aa_PC, aa_SCoM

def cal_PDBOX(seqlist, PDB_chain_dir, OX_DF_dir):
    
    if not os.path.exists(OX_DF_dir):
        os.msdir(OX_DF_dir)
    if os.path.exists(OX_DF_dir + '/PDB_X.pkl') and os.path.exists(OX_DF_dir + '/PDB_O.pkl'):
        print("File already exists!")
        return
    X_dict = {}
    O_dict = {}
    for pdb in tqdm(seqlist):
        file_path = PDB_chain_dir + '/{}.pdb'.format(pdb)
        chains = load_chains(pdb_id=pdb, file=PDB_chain_dir + '%s.pdb' %pdb)
        atom_coordinates, atom_ids = process_chain(chains)
        X, O = get_aa_frame(atom_coordinates, atom_ids, pdb)                     
        X_dict[pdb] = X
        O_dict[pdb] = O
    with open(OX_DF_dir + '/PDB_X.pkl', 'wb') as f:
        pkl.dump(X_dict, f)
    with open(OX_DF_dir + '/PDB_O.pkl', 'wb') as f:
        pkl.dump(O_dict, f)

def cal_DSSP(seq_list,dssp_dir,feature_dir):

    maxASA = {'G':188,'A':198,'V':220,'I':233,'L':304,'F':272,'P':203,'M':262,'W':317,'C':201,
              'S':234,'T':215,'N':254,'Q':259,'Y':304,'H':258,'D':236,'E':262,'K':317,'R':319}
    map_ss_8 = {' ':[1,0,0,0,0,0,0,0],'S':[0,1,0,0,0,0,0,0],'T':[0,0,1,0,0,0,0,0],'H':[0,0,0,1,0,0,0,0],
                'G':[0,0,0,0,1,0,0,0],'I':[0,0,0,0,0,1,0,0],'E':[0,0,0,0,0,0,1,0],'B':[0,0,0,0,0,0,0,1]}
    dssp_dict = {}
    if os.path.exists(feature_dir + 'DSSP.pkl'):
        return
    for seqid in seq_list:
        file = seqid+'.dssp'
        with open(dssp_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
        seq_feature = {}
        for i in range(28, len(fin_data)):
            line = fin_data[i]
            if line[13] not in maxASA.keys() or line[9]==' ':
                continue
            res_id = float(line[5:10])
            feature = np.zeros([14])
            feature[:8] = map_ss_8[line[16]] #Mapping secondary structure flags to 8-dimensional binary features
            feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1) #Calculate the characteristics of the relative solvent accessible surface area
            feature[9] = (float(line[85:91]) + 1) / 2 
            feature[10] = min(1, float(line[91:97]) / 180)
            feature[11] = min(1, (float(line[97:103]) + 180) / 360)
            feature[12] = min(1, (float(line[103:109]) + 180) / 360)
            feature[13] = min(1, (float(line[109:115]) + 180) / 360)
            seq_feature[res_id] = feature.reshape((1, -1))
        dssp_dict[file.split('.')[0]] = seq_feature
    with open(feature_dir + 'DSSP.pkl', 'wb') as f:
        pkl.dump(dssp_dict, f)


        
def cal_PSSM(seq_list,pssm_dir,feature_dir):
    if os.path.exists(feature_dir + 'PSSM.pkl'):
        print("File already exists!")
        return
    nor_pssm_dict = {}
    for seqid in tqdm(seq_list):
        file = seqid+'.pssm'
        with open(pssm_dir+'/'+file,'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for i in range(1,len(fin_data)):
                if fin_data[i] == '\n':
                    pssm_end_line = i
                    continue
            if pssm_end_line < pssm_begin_line:
                print(seqid)
                break
            feature = np.zeros([(pssm_end_line-pssm_begin_line),20])
            axis_x = 0
            for i in range(pssm_begin_line,pssm_end_line):
                raw_pssm = fin_data[i].split()[2:22]
                axis_y = 0
                for j in raw_pssm:
                    feature[axis_x][axis_y]= (1 / (1 + math.exp(-float(j))))
                    axis_y+=1
                axis_x+=1
            nor_pssm_dict[file.split('.')[0]] = feature
    with open(feature_dir+'PSSM.pkl','wb') as f:
        pkl.dump(nor_pssm_dict,f)


def PDBResidueFeature(seqlist,PDB_DF_dir,feature_dir,residue_feature_list,feature_combine,atomfea):
    if os.path.exists(feature_dir+'/'+'_residue_feas_'+feature_combine+'.pkl'):
        print("File already exists!")
        return
    for fea in residue_feature_list:
        with open(feature_dir + '{}.pkl'.format(fea), 'rb') as f:
            locals()['residue_fea_dict_' + fea] = pkl.load(f)

    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85,'H':1.2,'D':1.2,'SE':1.9,'P':1.8,'FE':2.23,'BR':1.95,
                        'F':1.47,'CO':2.23,'V':2.29,'I':1.98,'CL':1.75,'CA':2.81,'B':2.13,'ZN':2.29,'MG':1.73,'NA':2.27,
                        'HG':1.7,'MN':2.24,'K':2.75,'AC':3.08,'AL':2.51,'W':2.39,'NI':2.22}
    #Radius of van der Waals
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    residue_feas_dict = {}
    for seq_id in tqdm(seqlist):
        # print(seq_id)
        with open(PDB_DF_dir+'/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pkl.load(f)

        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        pdb_res_i = pdb_res_i[pdb_res_i['atom_type']!='H']
        mass = np.array(pdb_res_i['mass'].tolist()).reshape(-1, 1)
        mass = mass / 32

        B_factor = np.array(pdb_res_i['B_factor'].tolist()).reshape(-1, 1)
        if (max(B_factor) - min(B_factor)) == 0:
            B_factor = np.zeros(B_factor.shape) + 0.5
        else:
            B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
        is_sidechain = np.array(pdb_res_i['is_sidechain'].tolist()).reshape(-1, 1)
        occupancy = np.array(pdb_res_i['occupancy'].tolist()).reshape(-1, 1)
        charge = np.array(pdb_res_i['charge'].tolist()).reshape(-1, 1)
        num_H = np.array(pdb_res_i['num_H'].tolist()).reshape(-1, 1)
        ring = np.array(pdb_res_i['ring'].tolist()).reshape(-1, 1)

        atom_type = pdb_res_i['atom_type'].tolist()
        atom_vander = np.zeros((len(atom_type), 1))
        for i, type in enumerate(atom_type):
            try:
                atom_vander[i] = atom_vander_dict[type]
            except:
                atom_vander[i] = atom_vander_dict['C']

        atom_feas = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]
        atom_feas = np.concatenate(atom_feas,axis=1)
        residue_feas = []
        for fea in residue_feature_list:
            fea_i = locals()['residue_fea_dict_' + fea][seq_id]
            if isinstance(fea_i, np.ndarray):
                residue_feas.append(fea_i)
            elif isinstance(fea_i, dict):
                fea_ii = []
                for res_id_i in res_id_list:
                    if res_id_i in fea_i.keys():
                        fea_ii.append(fea_i[res_id_i])
                    else:
                        fea_ii.append(np.zeros(list(fea_i.values())[0].shape))
                fea_ii = np.concatenate(fea_ii,axis=0)
                residue_feas.append(fea_ii)
        try:
            residue_feas = np.concatenate(residue_feas, axis=1)
        except ValueError:
            print('Protein {} atomic feature dimension mismatch!'.format(seq_id))
            continue
        if residue_feas.shape[0] != len(res_id_list):
            print('Atomic characterization of protein {} residues is missing!'.format(seq_id))
            raise IndexError

        if atomfea:
            res_atom_feas = []
            atom_begin = 0
            for i, res_id in enumerate(res_id_list):

                res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
                atom_num = len(res_atom_df)
                res_atom_feas_i = atom_feas[atom_begin:atom_begin+atom_num]
                res_atom_feas_i = np.average(res_atom_feas_i,axis=0).reshape(1,-1)
                res_atom_feas.append(res_atom_feas_i)
                atom_begin += atom_num
            res_atom_feas = np.concatenate(res_atom_feas,axis=0)
            residue_feas = np.concatenate((res_atom_feas,residue_feas),axis=1)

        residue_feas_dict[seq_id] = residue_feas

    with open(feature_dir+'/'+'_residue_feas_'+feature_combine+'.pkl', 'wb') as f:
        pkl.dump(residue_feas_dict, f)

    
def prepare_features(pdb, pkl_dir, save_dir, max_len, NODE_DIM):

    with open(pkl_dir+"PDB_X.pkl", "rb") as f:
        X = pkl.load(f)[pdb]
    padded_X = np.zeros((max_len, 3))
    padded_X[:X.shape[0]] = X
    padded_X = torch.tensor(padded_X, dtype = torch.float)

    with open(pkl_dir+"PDB_O.pkl", "rb") as f:
        O = pkl.load(f)[pdb]
    padded_O = np.zeros((max_len,3,3))    
    padded_O[:O.shape[0]] = O
    padded_O = torch.tensor(padded_O, dtype = torch.float)
    
    with open(pkl_dir+"_residue_feas_PSA.pkl", "rb") as f:
        node_features = pkl.load(f)[pdb]
    padded_node_features = np.zeros((max_len, NODE_DIM))
    padded_node_features[:node_features.shape[0]] = node_features
    padded_node_features = torch.tensor(padded_node_features, dtype = torch.float)

    masks = np.zeros(max_len)
    masks[:X.shape[0]] = 1
    masks = torch.tensor(masks, dtype = torch.long)


    # Save
    torch.save(padded_X, save_dir + f'{pdb}_X.tensor')
    torch.save(padded_O, save_dir + f'{pdb}_O.tensor')
    torch.save(padded_node_features, save_dir + f'{pdb}_node_feature.tensor')
    torch.save(masks, save_dir + f'{pdb}_mask.tensor')
    # torch.save(padded_y, save_dir + f'{pdb_id}_label.tensor')