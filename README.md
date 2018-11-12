# inhibitor_prediction
Course project for CS 273B and CS 229.  The goal is to train a deep learning model that takes one of eight kinases and a potential inhibitor, and predicts whether or not the kinase will be inhibited

# Setup
1. Download the dataset from https://www.kaggle.com/xiaotawkaggle/inhibitors/home
2. Unzip and rename to "data", place in the repo's base directory
3. Create a conda virtual environment https://conda.io/docs/user-guide/tasks/manage-environments.html
    4. conda create -n myenv python=3.5
    5. activate
6. conda install cython
7. conda install -c conda-forge mdtraj
8. Install rdkit https://www.rdkit.org/docs/Install.html
9. gpu=0 bash scripts/install_deepchem_conda.sh (gpu=1 for GPU support)
