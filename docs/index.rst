.. actsnclass documentation master file, created by
   sphinx-quickstart on Thu Aug  8 16:11:41 2019.



Welcome to actsnclass !
=======================



This tool allows you to reproduce the results presented in `Ishida et al., 2019 <https://cosmostatistics-initiative.org/portfolio-item/active-learning-for-sn-classification/>`_.
It was based on the original prototype developed during the `COIN Residence Program #4 <http://iaacoin.wixsite.com/crp2017/>`_ , which took place in Clermont Ferrand, France, August 2017.


The code has been updated to allow the inclusion of different modules and facilitate contributions.

Analysis steps
==============

.. image:: images/active_learning_loop.png
   :align: right
   :height: 255.6px
   :width: 225 px
   :alt: Active learning loop

The ``actsnclass`` pipeline is composed of 4 different modules:

1. :ref:`Feature extraction <featext>`

2. :ref:`Classifier <classifier>`

3. Learning Strategy

4. Metric evaluation


These are arranged in the adaptable learning process (figure to the right).


Dependencies
============

``actsnclass`` was developed under ``Python3``. The complete list of dependencies is given below:

 - Python>=3.6.8
 - argparse>=1.1
 - matplotlib>=3.1.1
 - numpy>=1.17.0
 - pandas>=0.25.0
 - setuptools>=41.0.1
 - scipy>=1.3.0
 - sklearn>=0.20.3


Installing
==========

Clone this repository and type::

    >> python setup.py install --user


This will install ``actsnclass`` tools under your home.

Table of Contents
=================

.. contents:: :local:

.. toctree::
   :maxdepth: 2

   feature_extraction
   reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`