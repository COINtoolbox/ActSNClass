# <img align="right" src="https://github.com/COINtoolbox/ActSNClass/blob/master/images/COIN_logo_very_small.png" width="350"> Active Learning for Supernova Photometric Classification 

This repository holds the code and data used in [Optimizing spectroscopic follow-up strategies for supernova photometric classification with active learning](https://arxiv.org/abs/1804.03765), by [Ishida](https://www.emilleishida.com), [Beck](https://github.com/beckrob), [Gonzalez-Gaitan](https://centra.tecnico.ulisboa.pt/team/?id=4337), [de Souza](https://www.rafaelsdesouza.com), [Krone-Martins](https://thegrid.ai/alberto-krone-martins/), [Barrett](http://jimbarrett.co.uk/), [Kennamer](https://github.com/NobleKennamer), [Vilalta](http://www2.cs.uh.edu/~vilalta/), [Burgess](https://grburgess.github.io/), [Quint](https://github.com/b1quint), [Vitorelli](https://github.com/andrevitorelli), [Mahabal](http://www.astro.caltech.edu/~aam/) and [Gangler](https://annuaire.in2p3.fr/agents/Y249R2FuZ2xlciBFbW1hbnVlbCxvdT1wZW9wbGUsZGM9aW4ycDMsZGM9ZnI=/show), 2018.

This is one of the products of [COIN Residence Program #4](http://iaacoin.wix.com/crp2017), which took place in August/2017 in Clermont-Ferrand (France). 

We kindly ask you to include the full citation if you use this material in your research: [Ishida et al., 2018 -  arXiv:astro-ph/1804.03765](https://arxiv.org/abs/1804.03765).


# Pipeline

The `examples` folder contains all the necessary steps to perform the basic steps in the pipeline:

1. Light cure fit using the parametric function from [Bazin et al., 2009](https://arxiv.org/abs/0904.1066)
2. Separate the photometric sample in query/target
3. Get the simulated features used to identify neighbors
4. Build pseudo-training sample for the canonical strategy
5. Run the active learning algorithm - we use the Active Learning implementation in [libact](https://github.com/ntucllab/libact)


# Install 

The current version runs in Python-2.7.

In order to install this code you should clone this repository and do:  

        python setup.py install --user

# Dependencies

## General

    - Python=2.7
    - numpy>=1.8.2
    - matplotlib>=1.3.1
    - pandas>=0.19.2
    - scipy>=0.18.1
    - astropy>=2.0.1
    - libact>=0.1.3
    - gptools>=0.2.3
    - cython>=0.29.6
    - seaborn>=0.9.0

## For pre-processing

You only need to install this package if you intend to use the SNANA file handling as shown in the `examples` folder. 

If you already have your data reduced you can skip this part.


[snclass](https://github.com/emilleishida/snclass)



