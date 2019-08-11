.. actsnclass documentation master file, created by
   sphinx-quickstart on Thu Aug  8 16:11:41 2019.



Welcome to actsnclass !
=======================



This tool allows you to reproduce the results presented in `Ishida et al., 2019 <https://cosmostatistics-initiative.org/portfolio-item/active-learning-for-sn-classification/>`_.
It was based on the original prototype developed during the `COIN Residence Program #4 <http://iaacoin.wixsite.com/crp2017/>`_ , which took place in Clermont Ferrand, France, August 2017.


The code has been updated to allow a friendly use and expansion.

Getting started
===============

In order to setup a suitable working environment, clone this repository and make sure you have the necessary packages installed.

Dependencies
------------

``actsnclass`` was developed under ``Python3``. The complete list of dependencies is given below:

 - Python>=3.6.8
 - argparse>=1.1
 - matplotlib>=3.1.1
 - numpy>=1.17.0
 - pandas>=0.25.0
 - setuptools>=41.0.1
 - scipy>=1.3.0
 - sklearn>=0.20.3
 - seaborn>=0.9.0

Installing
----------

Clone this repository,

.. code-block:: bash

    >>> git clone https://github.com/COINtoolbox/ActSNClass

navigate to where you stored it and install it:

.. code-block:: bash

    >> python setup.py install --user

This will install ``actsnclass`` tools under your home.

Setting up a working directory
------------------------------

In another location of your choosing, create the following directory structure:

::

    work_dir
    ├── plots
    ├── results

The outputs of ``actsnclass`` will be stored in these directories.

In order to set things properly, from the repository you just cloned, and move the data directory to your
chosen working directory and unpack the data.

.. code-block:: bash

    >>> mv -rf actsnclass/data/ work_dir/
    >>> cd work_dir/data
    >>> tar -xzvf SIMGEN_PUBLIC_DES.tar.gz

This data was provided by Rick Kessler, after the publication of results from the
`SuperNova Photometric Classification Challenge <https://arxiv.org/abs/1008.1024>`_.


Analysis steps
==============

.. image:: images/active_learning_loop.png
   :align: right
   :height: 255.6px
   :width: 225 px
   :alt: Active learning loop

The ``actsnclass`` pipeline is composed of 4 important steps:

1. Feature extraction

2. Classifier

3. Query Strategy

4. Metric evaluation


These are arranged in the adaptable learning process (figure to the right).

Using this package
------------------

Step 1 is considered pre-processing. The current code does the feature extraction
using the `Bazin parametric function <https://arxiv.org/abs/0904.1066>`_ for the complete training and test sample
before any machine learning application is used.

Details of the tools available to evaluate different steps on feature extraction can be found in the
:ref:`Feature extraction page <preprocessing>`.

Alternatively, you can also perform the full light curve fit for the entire sample from the command line:

.. code-block:: bash

    >>> fit_dataset.py -dd <path_to_data_dir> -o <output_file>



Table of Contents
=================

.. contents:: :local:

.. toctree::
   :maxdepth: 2

   pre_processing
   learn_loop
   reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`