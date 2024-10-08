#!/usr/bin/env python3
import os
import sys
import argparse
import warnings
from tqdm import tqdm
from FileConversion import PDBtoDSSP, PDBtoFASTA, FASTAtoPSSM, PDBtoCSV
from DataPreprocessing import cal_PDBOX, cal_DSSP, cal_PSSM, PDBResidueFeature, prepare_features
from utils import extract_prefixes_from_folder

warnings.filterwarnings("ignore")

def main(data_dir, maxlen=2000):
    # Define folders
    PDB_folder = os.path.join(data_dir, "PDB/")
    DSSP_folder = os.path.join(data_dir, "DSSP/")
    FASTA_folder = os.path.join(data_dir, "FASTA/")
    PSSM_folder = os.path.join(data_dir, "PSSM/")
    CSV_folder = os.path.join(data_dir, "CSV/")
    Tensor_folder = os.path.join(data_dir, 'Tensor/')
    nodedim = 41

    # Create output folders if they don't exist
    for folder in [DSSP_folder, FASTA_folder, PSSM_folder, CSV_folder, Tensor_folder]:
        os.makedirs(folder, exist_ok=True)

    # Get list of PDB files
    fileLine = os.listdir(PDB_folder)
    seqList = [x.split('.')[0] for x in fileLine]
    seqlist = [s for s in seqList if s]

    # Step 1: File Conversion
    print('File Conversion:\n1. Converting PDB files into DSSP files...')
    PDBtoDSSP(PDB_folder, DSSP_folder)

    print('2. Converting PDB files into FASTA files...')
    PDBtoFASTA(PDB_folder, FASTA_folder)

    print('3. Converting FASTA files into PSSM files...')
    FASTAtoPSSM(FASTA_folder, PSSM_folder)

    print('4. Converting PDB files into CSV files...')
    PDBtoCSV(PDB_folder, CSV_folder)

    # Step 2: Data Preprocessing
    feature_list = ['PSSM', 'DSSP']
    feature_combine = 'PSA'
    atomfea = True

    print('Data Preprocessing:\n1. Extracting spatial information...')
    cal_PDBOX(seqlist, PDB_folder, data_dir)

    print('2. Extracting secondary structure information...')
    cal_DSSP(seqlist, DSSP_folder, data_dir)

    print('3. Extracting genetic information...')
    cal_PSSM(seqlist, PSSM_folder, data_dir)

    print('4. Computing atomic information and combining node attributes...')
    PDBResidueFeature(seqlist, CSV_folder, data_dir, feature_list, feature_combine, atomfea)

    # Step 3: Feature Preparation
    print('Preparing features...')
    for pdb in tqdm(seqlist):
        if f'{pdb}_X.tensor' not in os.listdir(Tensor_folder):
            prepare_features(pdb, data_dir, Tensor_folder, maxlen, nodedim)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Protein Data Processing and Feature Extraction")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing the PDB data")
    parser.add_argument('--maxlen', type=int, default=2000, help="Maximum length for protein sequences (default: 2000)")

    args = parser.parse_args()

    # Execute main function with parsed arguments
    main(args.data_dir, args.maxlen)