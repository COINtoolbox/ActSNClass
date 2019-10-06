.. _contribute:

How to contribute
=================

Below you will find general guidance on how to prepare your piece of code to be integrated to the
``actsnclass`` environment.


Add a new data set
------------------

The main challenge of adding a new data set is to build the infrastructure necessary to handle the new data.

The function below show how the basic structure required to deal with 1 light curve:

.. code-block:: python
   :linenos:

   >>> import pandas as pd

   >>> def load_one_lightcurve(path_to_data, *args):
   >>>     """Load 1 light curve at a time.
   >>>
   >>>     Parameters
   >>>     ----------
   >>>     path_to_data: str
   >>>         Complete path to data file.
   >>>     ...
   >>>         ...
   >>>
   >>>     Returns
   >>>     -------
   >>>     pd.DataFrame
   >>>     """
   >>>
   >>>    ####################
   >>>    # Do something #####
   >>>    ####################
   >>>
   >>>    # structure of light curve
   >>>    lc = {}
   >>>    lc['dataset_name'] = XXXX               # name of the data set
   >>>    lc['filters'] = [X, Y, Z]               # list of filters
   >>>    lc['id'] = XXX                          # identification number
   >>>    lc['redshift'] = X                      # redshift (optional, important for building canonical)
   >>>    lc['sample'] = XXXXX                    # train, test or queryable (none is mandatory)
   >>>    lc['sntype'] = X                        # Ia or non-Ia
   >>>    lc['photometry' = pd.DataFrame()        # min keys: MJD, filter, FLUX, FLUXERR
   >>>                                            # bonus: MAG, MAGERR, SNR
   >>>    return lc

Feel free to also provide other keywords which might be important to handle your data.
Given a function like this we should be capable of incorporating it into the pipeline.

Please refer to the :py:mod:`actsnclass.fit_lightcurves` module for a closer look at this part of the code.



Add a new feature extraction method
-----------------------------------

Currently ``actsnclass`` only deals with Bazin features.
The snipet below show an example of friendly code for a new feature extraction method.


.. code-block:: python
   :linenos:

   >>> def new_feature_extraction_method(time, flux, *args):
   >>>    """Extract features from light curve.
   >>>
   >>>    Parameters
   >>>    ----------
   >>>    time: 1D - np.array
   >>>        Time of observation.
   >>>    flux: 1D - np.array of floats
   >>>        Measured flux.
   >>>    ...
   >>>        ...
   >>>
   >>>    Returns
   >>>    -------
   >>>    set of features
   >>>    """
   >>>
   >>>         ################################
   >>>         ###   Do something    ##########
   >>>         ################################
   >>>
   >>>    return features

You can check the current feature extraction tools for the Bazin parametrization at :py:mod:`actsnclass.bazin`
module.


Add a new classifier
--------------------

A new classifier should be warp in a function such as:

.. code-block:: python
   :linenos:

   >>> def new_classifier(train_features, train_labels, test_features, *args):
   >>>     """Random Forest classifier.
   >>>
   >>>     Parameters
   >>>     ----------
   >>>     train_features: np.array
   >>>         Training sample features.
   >>>     train_labels: np.array
   >>>         Training sample classes.
   >>>     test_features: np.array
   >>>         Test sample features.
   >>>     ...
   >>>         ...
   >>>
   >>>    Returns
   >>>     -------
   >>>     predictions: np.array
   >>>         Predicted classes - 1 class per object.
   >>>     probabilities: np.array
   >>>         Classification probability for all objects, [pIa, pnon-Ia].
   >>>     """
   >>>
   >>>    #######################################
   >>>    #######  Do something     #############
   >>>    #######################################
   >>>
   >>>    return predictions, probabilities

The only classifier implemented at this point is a Random Forest and can be found at the
:py:mod:`actsnclass.classifiers` module.

.. important:: Remember that in order to be effective in the active learning frame work a classifier should not be heavy on the required computational resources and must be sensitive to small changes in the training sample. Otherwise the evolution will be difficult to tackle.

Add a new query strategy
------------------------

A query strategy is a protocol which evaluates the current state of the machine learning model and
makes an informed decision about which objects should be included in the training sample.

This is very general, and the function can receive as input any information regarding the physical
properties of the test and/or target samples and current classification results.

A minimum structure for such function would be:

.. code-block:: python
   :linenos:

   >>> def new_query_strategy(class_prob, test_ids, queryable_ids, batch, *args):
   >>>     """New query strategy.
   >>>
   >>>     Parameters
   >>>     ----------
   >>>     class_prob: np.array
   >>>         Classification probability. One value per class per object.
   >>>     test_ids: np.array
   >>>         Set of ids for objects in the test sample.
   >>>     queryable_ids: np.array
   >>>         Set of ids for objects available for querying.
   >>>     batch: int
   >>>         Number of objects to be chosen in each batch query.
   >>>     ...
   >>>         ...
   >>>
   >>>     Returns
   >>>     -------
   >>>     query_indx: list
   >>>         List of indexes identifying the objects from the test sample
   >>>         to be queried in decreasing order of importance.
   >>>     """
   >>>
   >>>        ############################################
   >>>        #####   Do something              ##########
   >>>        ############################################
   >>>
   >>>     return list of indexes of size batch


The current available strategies are Passive Learning (or Random Sampling) and Uncertainty Sampling.
Both can be scrutinized at the :py:mod:actsnclass.`query_strategies` module.


Add a new diagnostic metric
---------------------------

Beyond the criteria for choosing an object to be queried one could also think about the possibility
to test different metrics to evaluate the performance of the classifier at each learning loop.

A new diagnostic metrics can then be provided in the form:

.. code-block:: python
   :linenos:

   >>> def new_metric(label_pred: list, label_true: list, ia_flag, *args):
   >>>     """Calculate efficiency.
   >>>
   >>>     Parameters
   >>>     ----------
   >>>     label_pred: list
   >>>         Predicted labels
   >>>     label_true: list
   >>>         True labels
   >>>     ia_flag: number, symbol
   >>>         Flag used to identify Ia objects.
   >>>     ...
   >>>         ...
   >>>
   >>>     Returns
   >>>     -------
   >>>     a number or set of numbers
   >>>         Tells us how good the fit was.
   >>>     """
   >>>
   >>>     ###########################################
   >>>     #####  Do something !    ##################
   >>>     ###########################################
   >>>
   >>>     return a number or set of numbers

The currently implemented diagnostic metrics are those used in the
SNPCC (`Kessler et al., 2009 <https://arxiv.org/abs/1008.1024>`_) and can be found at the
:py:mod:`actsnclass.metrics` module.

