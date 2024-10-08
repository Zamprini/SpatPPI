## Installation and implementation of SpatPPI

### 1. Description

SpatPPI is a deep learning-based model that predicts protein-protein interactions (PPIs) by generating detailed feature representations for each amino acid. It utilizes 3D protein structures predicted by AlphaFold2 and converts them into directed graphs, where nodes represent amino acid residues and edges capture their spatial relationships. Nodes are encoded with attributes reflecting evolutionary information, protein secondary structures, and atomic chemical properties. SpatPPI builds a local coordinate system for each residue and uses seven-dimensional edge attributes to describe spatial relationships, including 3D coordinates and rotation quaternions. It employs a customized graph self-attention network to update node and edge attributes iteratively, integrating local spatial information into residue representations. Within a Siamese network framework, SpatPPI generates residue representations for protein pairs, uses a bilinear function to learn binding weights between residue embeddings, and produces a residue contact probability matrix to capture residue contact patterns. The model assesses the likelihood of residue conformational combinations to infer PPIs and is designed to account for structural variability of intrinsically disordered regions (IDRs), improving predictive performance for IDR-containing PPIs.

### 2. Installation

#### 2.1 system requirements

For prediction process, you can predict whether a pair of proteins interact with each other in milliseconds using only the CPU. However, for training a new deep model form scratch, we recommend using a GPU for significantly faster training (Training 50 generations on a dataset of 30,000 sample pairs took 1 day on a single A100 GPU) .

To use GeoNet with GPUs, you will need: cuda >= 10.0, cuDNN.

### 2.2 create an environment

SpatPPI is built on Python3.8. We highly recommend to use a virtual environment for the installation of SpatPPI and its dependencies.

A virtual environment can be created and (de)activated as follows by using conda(https://conda.io/docs/):

```cmd
# create
conda create -n SpatPPI python=3.8
# activate
conda activate SpatPPI
# deactivate
conda deactivate
```

### 2.3 Install SpatPPI dependencies

​    Note: If you are using a Python virtual environment, make sure it is activated before running each command in this guide.

#### 2.3.1 Install requirements

 	(1) Install pytorch 2.2.2 (For more details, please refer to https://pytorch.org/)

```cmd
# CUDA 11.8
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cpuonly -c pytorch    
```

​	(2) Install other requirements

```cmd
pip install -r requirements.txt
```

#### 2.3.2 Install the predicted (real) protein structure

You can batch download protein predicted structures in AlphaFold Protein Structure Database (https://alphafold.com/) or find real protein structures from the uniprot database (https://www.uniprot.org/).

#### 2.3.3 Install the bioinformatics tools

​	(1) Install blast+ for extracting PSSM (position-specific scoring matrix) profiles

To install ncbi-blast-2.14.0+ or latest version and download NR database (ftp://ftp.ncbi.nlm.nih.gov/blast/db/) for psiblast, please refer to BLAST® Help (https://www.ncbi.nlm.nih.gov/books/NBK52640/).

```
tar zxvpf ncbi-blast-2.14.0.tar.gz
```

Set the absolute paths of blast+ and NR databases in the "Dataio/FileConversion.py"

​	(2) Install DSSP for extracting SS (Secondary structure) profiles

### 3. Usage

#### 3.1 Batch prediction of protein-protein interactions

Example：

```cmd
# Generate tensor files needed for model input based on PDB files
cd Dataio
python protein_processing.py --data_dir ./ --maxlen 2000
--------------------------------------------------------------------
The list of commands:
    --data_dir         The PDB files to be predicted are placed in a PDB folder, enter the folder path here.
    --maxlen           The maximum number of residuals acceptable to the pre-trained model you are using(Default is 2000).
--------------------------------------------------------------------
#Batch prediction of protein-protein interactions (based on a csv file)
cd ..
cd ModelCode
python prediction.py --data_folder ../Datasets/Example/ --model_path ../Models/HuRI-IDP/SpatPPI.pth --predicted_filename Example_Test.csv
--------------------------------------------------------------------
The list of commands:
    --data_folder          Path to the data folder containing CSV files and tensors
    --model_path           Path to the saved model (.pth) file
    --predicted_filename   The filename of the test dataset under data_folder(csv file)
```

#### 3.2 Train a new deep model from scratch

Example

```cmd
#Assuming all protein Tensor files are ready
cd ModelCode
python training.py --data_folder ../Datasets/Example/ --train_filename Example_Train.csv --test_filename Example_Test.csv
--------------------------------------------------------------------
The list of commands:
    --data_folder      Path to the data folder containing CSV files and tensors
    --train_filename   The filename of the train dataset under data_folder(csv file)
    --test_filename    The filename of the test dataset under data_folder(csv file)
```

The parameters of the model and the address where the training model and results are saved can be set in the **configs.py** file.

#### 3.3 Use of online services

You can use SpatPPI online by visiting http://liulab.top/SpatPPI/server.

### 4 How to cite SpatPPI?

If you are using the SpatPPI program, you can cite: