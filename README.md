[![crp4](https://img.shields.io/badge/CRP-%234-blue)](https://iaacoin.wixsite.com/crp2017)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-10.1093%2Fmnras%2Fsty3015-green)](https://doi.org/10.1093/mnras/sty3015) 
[![arXiv](https://img.shields.io/badge/arxiv-astro--ph%2F1804.03765-red)](https://arxiv.org/abs/1804.03765) 


# <img align="right" src="docs/images/COIN_logo_very_small.png" width="350"> ActSNClass


## Active Learning for Supernova Photometric Classification 

This repository holds the code and data used in [Optimizing spectroscopic follow-up strategies for supernova photometric classification with active learning](https://arxiv.org/abs/1804.03765), by [Ishida](https://www.emilleishida.com), [Beck](https://github.com/beckrob), [Gonzalez-Gaitan](https://centra.tecnico.ulisboa.pt/team/?id=4337), [de Souza](https://www.rafaelsdesouza.com), [Krone-Martins](https://thegrid.ai/alberto-krone-martins/), [Barrett](http://jimbarrett.co.uk/), [Kennamer](https://github.com/NobleKennamer), [Vilalta](http://www2.cs.uh.edu/~vilalta/), [Burgess](https://grburgess.github.io/), [Quint](https://github.com/b1quint), [Vitorelli](https://github.com/andrevitorelli), [Mahabal](http://www.astro.caltech.edu/~aam/) and [Gangler](https://annuaire.in2p3.fr/agents/Y249R2FuZ2xlciBFbW1hbnVlbCxvdT1wZW9wbGUsZGM9aW4ycDMsZGM9ZnI=/show), 2018.

This is one of the products of [COIN Residence Program #4](http://iaacoin.wix.com/crp2017), which took place in August/2017 in Clermont-Ferrand (France). 

We kindly ask you to include the full citation if you use this material in your research: [Ishida et al, 2019, MNRAS, 483 (1), 2–18](https://cosmostatistics-initiative.org/wp-content/uploads/2019/06/COIN_ActSNClass.txt).

Full documentation can be found at [readthedocs](https://actsnclass.readthedocs.io/en/latest/index.html#).

# Dependencies

### For code:

 - Python>=3.7  
 - argparse>=1.1  
 - matplotlib>=3.1.1  
 - numpy>=1.17.0  
 - pandas>=0.25.0  
 - setuptools>=41.0.1  
 - scipy>=1.3.0
 - scikit-learn>=0.20.3
 - seaborn>=0.9.0
 - xgboost>=1.6.2
 
 
 ### For documentation:
 
  - sphinx>=2.1.2

# Install

The current version runs in Python-3.7 or latter.

We recommend you use a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to ensure the correct package versions.

Once your environment is created, you can source it :  
    >> source <path_to_venv>/bin/activate
    
You will notice a `(ActSNCLass)` to the left of your terminal line.
This means everything is ok!

In order to install this code you should clone this repository and do::  

    (ActSNClass) >> pip install --upgrade pip
    (ActSNClass) >> pip install -r requirements
    (ActSNClass) >> python setup.py install


# Mlflow Integration 

I have successfully integrated MLflow for tracking and model registry in the project. This integration is implemented in the classifier.py file.

### Function Name:
 
- mlflow_tracking_And_Registry: this function handles the tracking of model metrics, parameters, and artifacts with MLflow, and registers the trained model in the MLflow Model Registry.


### Enabling Mlflow

To utilize MLflow for tracking and model registry, simply set the mlflow parameter to True in the random_forest function, which is located in the classifier.py file.

- Example Usage: random_forest( Parameters, mlflow=True)


### Accessing the Mlflow User Interface 

Once you have run the code with MLflow enabled, you can access the MLflow user interface to monitor your experiments and models.

- Start the MLflow UI by running the following command in your terminal::

     (ActSNClass) >> mlflow ui 

- Visit the MLflow UI by opening your web browser and navigating to: http://127.0.0.1:5000 

### Reusing Models from MLflow Registry

To download and reuse models that have been registered in the MLflow Model Registry, you can refer to the example provided in the managemodel.py file located in the actsnclass directory
