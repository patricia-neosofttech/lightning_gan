## PyTorch Lightning based implementation for distributed multi-GPU training for synthetic data generation

## Objective:
- As of now training of NN on limited amount of data could be restricted to single GPU instance which can train quickly (couple of hours), but training on larger amounts of data needs multiple GPU's for faster training and also needs to be utilized fully. 
- To perform multi-GPU training, we must have a way to split the model and data across different GPU's and to coordinate the training using pytorch lightning module.

## Setting up

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

-  Create new project directory.
 ```bash
 mkdir  project_name
 ```
-  cd to this project directory
```bash
cd project_name
```
- create training_weight & plots folder inside project directory to save weights & plot of image
 ```bash
 mkdir plots
 ```
- create the virtual environment.
```bash
python -m venv venv
```

-  activate your virtualenv
```bash
source venv/bin/activate
```
-  run below command in your shell.
```bash
pip install -r requirements.txt
```
## How to run the code
- Execute the below command to train the network
```bash
python main.py 

```
