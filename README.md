# inhibitor_prediction
Course project for CS 273B and CS 229.  The goal is to train a deep learning model that takes one of eight kinases and a potential inhibitor, and predicts whether or not the kinase will be inhibited

# Setup
1. Download the dataset from https://www.kaggle.com/xiaotawkaggle/inhibitors/home
2. Unzip and rename to "data", place in the repo's base directory
3. Create a virtualenv, and activate
4. Run `pip install -r requirements.txt`
5. Install rdkit https://www.rdkit.org/docs/Install.html
6. Install deepchem https://github.com/deepchem/deepchem
