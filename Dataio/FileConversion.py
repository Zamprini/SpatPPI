import os
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

amino_acid_mapping = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
                      'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                      'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

def PDBtoDSSP(PDB_folder, DSSP_folder):
    if not os.path.exists(DSSP_folder):
        os.makedirs(DSSP_folder)
        
    fileLine = os.listdir(PDB_folder)
    for i in tqdm(range(len(fileLine))):
        input_file = PDB_folder + fileLine[i]
        filename = fileLine[i].split('.')[0]+'.dssp'
        output_file = DSSP_folder + filename
        if os.path.exists(output_file):
            continue
        os.system('cd /root/miniconda3/bin/')
        os.system('mkdssp -i %s -o %s' %(input_file,output_file))   

        
def PDBtoFASTA(PDB_folder, FASTA_folder):
    if not os.path.exists(FASTA_folder):
        os.makedirs(FASTA_folder)
    
    fileLine = os.listdir(PDB_folder)
    for i in tqdm(range(len(fileLine))): 
        input_file = PDB_folder + fileLine[i]
        filename = fileLine[i].split('.')[0]+'.fasta'
        output_file = FASTA_folder + filename
        if os.path.exists(output_file):
            continue
        sequence = ""
        processed_residues = set()  # Used to record processed amino acid identifiers
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    # Get amino acid identifiers, including chain identifiers and residue numbers
                    residue_identifier = line[21:26].strip()
                    if residue_identifier in processed_residues:
                        continue
                    amino_acid_code = line[17:20].strip()
                    # Convert three-letter codes to single-character codes using the provided mapping
                    single_letter_code = amino_acid_mapping.get(amino_acid_code, 'X')  # 'X'表示未知的代码
                    sequence += single_letter_code
                    processed_residues.add(residue_identifier)
        with open(output_file, 'w') as of:
            of.write(f">{filename}\n{sequence}\n")


                     
def FASTAtoPSSM(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for fasta_file_name in tqdm(os.listdir(input_folder)):
        if fasta_file_name.endswith(".fasta"):
            fasta_file = os.path.join(input_folder, fasta_file_name)
            output_file = os.path.join(output_folder, f"{os.path.splitext(fasta_file_name)[0]}.pssm")
            if os.path.exists(output_file):
                continue
            # Convert FASTA files to PSSM files using psiblast
            command = f'psiblast -query {fasta_file} -db database/swissprot -evalue 0.001 -num_iterations 3 -out_ascii_pssm {output_file_path}'
            os.system(command)

def def_atom_features():
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}
    #20 amino acids, dictionary bonds are atoms, and the values are triplets representing the inclusion charge, the number of hydrogen atoms, and whether or not they are cyclic
    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                   'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0]/2+0.5,i_fea[1]/3,i_fea[2]]
    return atom_features

def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict = amino_acid_mapping
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path,'r')
    pdb_res = pd.DataFrame(columns=['ID','atom','atom_type','res','res_id','xyz','occupancy','B_factor','mass',
                                   'is_sidechain','charge','num_H','ring'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H':1,'C':12,'O':16,'N':14,'S':32,'FE':56,'P':31,'BR':80,'F':19,'CO':59,'V':51,
                            'I':127,'CL':35.5,'CA':40,'B':10.8,'ZN':65.5,'MG':24.3,'NA':23,'HG':200.6,'MN':55,
                            'K':39.1,'AP':31,'AC':227,'AL':27,'W':183.9,'SE':79,'NI':58.7}
    #atomic mass
    k = 0
    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count+=1
            res_pdb_id = int(line[22:26])
            if res_pdb_id != before_res_pdb_id:
                res_count +=1
            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N','CA','C','O','H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5,0.5,0.5]
            tmps = pd.Series(
                {'ID': atom_count, 'atom':line[12:16].strip(),'atom_type':atom_type, 'res': res, 'res_id':k + int(line[22:26]),
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),'occupancy':float(line[54:60]),
                 'B_factor': float(line[60:66]),'mass':Relative_atomic_mass[atom_type],'is_sidechain':is_sidechain,
                 'charge':atom_fea[0],'num_H':atom_fea[1],'ring':atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != k + int(line[22:26]):
                res_id_list.append(k + int(line[22:26]))
            pdb_res = pdb_res._append(tmps,ignore_index=True)
        if line.startswith('TER'):
            k = res_id_list[-1]
            continue
        if line.startswith('END'):
            break

    return pdb_res,res_id_list

def PDBtoCSV(PDB_folder, CSV_folder):

    if not os.path.exists(CSV_folder):
        os.mkdir(CSV_folder)
    
    fileLine = os.listdir(PDB_folder)
    for i in tqdm(range(len(fileLine))): 
        file_path = PDB_folder + fileLine[i]
        filename = fileLine[i].split('.')[0]+'.csv.pkl'
        output_file = CSV_folder + filename
        if os.path.exists(output_file):
            continue
        with open(file_path, 'r') as f:
            text = f.readlines()
        if len(text) == 1:
            print("Empty pdb file appears")
        try:
            pdb_DF, res_id_list = get_pdb_DF(file_path)
            with open(output_file, 'wb') as f:
                pickle.dump({'pdb_DF': pdb_DF, 'res_id_list': res_id_list}, f)
        except KeyError:
            print('Warning: skipped due to recognition of unknown residues{}'.fileLine[i])
            pass  