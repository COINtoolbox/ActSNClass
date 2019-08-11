***************
Reference / API
***************

.. currentmodule:: actsnclass


Light Curves
============

*Processing a light curve*

.. autosummary::
   :toctree: api

   LightCurve
   LightCurve.load_snpcc_lc
   LightCurve.fit_bazin
   LightCurve.fit_bazin_all
   LightCurve.plot_bazin_fit

*Fitting an entire data set*

.. autosummary::
   :toctree: api

   fit_snpcc_bazin


DataBase
========

*Object upon which the learning process is performed*

.. autosummary::
   :toctree: api

    DataBase
    DataBase.load_bazin_features
    DataBase.load_features
    DataBase.build_samples
    DataBase.classify
    DataBase.evaluate_classification
    DataBase.make_query
    DataBase.update_samples
    DataBase.save_metrics
    DataBase.save_queried_sample


Classifiers
===========

.. autosummary::
   :toctree: api

   random_forest


Query strategies
================

.. autosummary::
   :toctree: api

   random_sampling
   uncertainty_sampling

Metrics
=======

*Individual metrics*

.. autosummary::
   :toctree: api

   accuracy
   efficiency
   purity
   fom


*Metrics agregated by category or use*

.. autosummary::
   :toctree: api

   get_snpcc_metric


Active Learning loop
====================

.. autosummary::
   :toctree: api

    learn_loop

Scripts
=======

.. autosummary::
   :toctree: api

   fit_dataset
   run_loop
