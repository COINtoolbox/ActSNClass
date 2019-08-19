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
 
 
 ### For documentation:
 
  - sphinx>=2.1.2

# Install

The current version runs in Python-3.7.

We recommend you use anaconda to create a suitable environment.

To set up the enviroment clone this repository, navigate to its location in the 
terminal and do::  

    >> conda env create -f environment.yml

Once the environment is created, activate it using::

    >> conda activate ActSNClass
    
You will notice a `(ActSNCLass)` to the left of your terminal line.
This means everything is ok!

In order to install this code you should clone this repository and do::  

    (ActSNClass) >> python setup.py install

