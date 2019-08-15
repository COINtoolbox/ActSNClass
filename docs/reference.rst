***************
Reference / API
***************

.. currentmodule:: actsnclass

Pre-processing
==============

Light curve analysis
-------------------------

*Performing feature extraction for 1 light curve*

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

*Basic light curve analysis tools*

.. autosummary::

   snactclass.bazin.bazin
   snactclass.bazin.err_func
   snactclass.bazin.fit_scipy

Canonical sample
================

*The Canonical object for holding the entire sample.*

.. autosummary::
   :toctree: api

   Canonical
   Canonical.snpcc_get_canonical_info
   Canonical.snpcc_identify_samples
   Canonical.find_neighbors

*Functions to populate the Canonical object*

.. autosummary::
   :toctree: api

   build_snpcc_canonical
   plot_snpcc_train_canonical

Build time domain data base
===========================

.. autosummary::
   :toctree: api

   SNPCCPhotometry
   SNPCCPhotometry.get_lim_mjds
   SNPCCPhotometry.create_daily_file
   SNPCCPhotometry.build_one_epoch


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

*Full light curve*

.. autosummary::
   :toctree: api

    learn_loop

*Time domain*

.. autosummary::
   :toctree: api

    get_original_training
    time_domain_loop.time_domain_loop


Plotting
========

.. autosummary::
   :toctree: api

    Canvas
    Canvas.load_diagnostics
    Canvas.set_plot_dimensions
    Canvas.plot_diagnostics

Scripts
=======

.. autosummary::

   build_canonical
   build_time_domain
   fit_dataset
   make_diagnostic_plots
   run_loop
   run_time_domain
