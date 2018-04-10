# <img align="right" src="https://github.com/COINtoolbox/ActSNClass/blob/master/images/COIN_logo_very_small.png" width="350"> Active Learning for Supernova Photometric Classification 

This repository holds the code and data used in *Optimizing spectroscopic follow-up strategies for supernova photometric classification with active learning*, by [Ishida](www.emilleishida.com), Beck, Gonzalez-Gaitan, de Souza, Krone-Martins, Barrett, Kennamer, Vilalta, Burgess, Quint, Vitorelli, Mahabal and Gangler, 2018.

This is one of the products of [COIN Residence Program #4](http://iaacoin.wix.com/crp2017), which took place in August/2017 in Clermont-Ferrand (France). 

# Pipeline

The `examples` folder contains all the necessary steps to perform the basic steps in the pipeline:

1. Light cure fit using the parametric function from [Bazin et al., 2009](https://arxiv.org/abs/0904.1066)
2. Separate the photometric sample in query/target
3. Get the simulated features used to identify neighbors
4. Build pseudo-training sample for the canonical strategy
5. Run the active learning algorithm


# Install 

In order to install this code you should clone this repository and do:  

        python setup.py install --user

# Dependencies

## General

    - numpy>=1.8.2
    - matplotlib>=1.3.1
    - pandas>=0.19.2
    - scipy>=0.18.1
    - astropy>=2.0.1
    - libact>=0.1.3

## For pre-processing

    - [snclass](https://github.com/emilleishida/snclass)



