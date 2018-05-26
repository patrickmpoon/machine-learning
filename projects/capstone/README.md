# README: Traffic Stop Outcome Predictor

## Documents

1. The project **proposal** is available at: https://github.com/patrickmpoon/machine-learning/blob/master/projects/capstone/proposal.pdf
1. The project **report** is located at: https://github.com/patrickmpoon/machine-learning/blob/master/projects/capstone/report.pdf


## Requirements

These instructions were tested and verified on a **Ubuntu 16.04** server.   Please adjust and modify if you are using a different operating system.


## Setup

1. Install conda if it is not already installed
1. Create a temporary conda virtual environment:
`conda create -n ppoon_capstone python=3`
1. Activate this new environment:
`source activate ppoon_capstone`
1. Create a temporary directory for this project:
`mkdir ~/Downloads/tmp`
1. `cd ~/Downloads/tmp`
1. To avoid having to download every project directory in this repository, you can use svn to pull down only the directory for this Capstone project.
    1. If you do not already have subversion installed, you can get it by running:
    `sudo apt-get install subversion`
    1. Pull down this project's folder:
    `svn checkout https://github.com/patrickmpoon/machine-learning/trunk/projects/capstone`
1. `cd capstone`
1. Install the project software dependencies:
`pip install -r requirements`
1. Start the Jupyter Notebook Server:
`jupyter notebook`


## Preprocessing

First, we will need to generate the training and testing sets for model fitting.  Different configurations were tested and are specified in the main project notebook

1. From the Jupyter Notebook Server Home tab, click on the "Preprocessing.ipynb" link.
1. `Kernel > Restart & Run All` to run all the code cells.

Once the runs complete, several files will have been created in the `data` directory with the format: `stage[X]-[train|test|.pkl` where X is the stage number.


## Main Project Notebook
1. From the Jupyter Notebook Server Home tab, click on the "Traffic_Stop_Outcome_Predictor.ipynb" link.
1. `Kernel > Restart & Run All` to run all the code cells.

The project report's contents were primarily generated from this notebook.

**Note:** Be sure to run the steps in the Preprocssing section above first as this notebook imports *.pkl files that are generated from that section.


## Implementation

1. From the Jupyter Notebook Server Home tab, individually run the  "Stage [X].ipynb" notebooks to train and test the classifiers at each stage.  The resulting scores are tabulated in the main project notebook.


## Refinement

1. From the Jupyter Notebook Server Home tab, click on the "Refinement.ipynb" link.
1. `Kernel > Restart & Run All` to run all the code cells to see the steps taken in the Refinement stage.
