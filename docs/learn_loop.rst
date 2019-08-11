.. _learnloop:

Active Learning loop
====================

Once the data has been pre-processed, analysis steps 2-4 can be performed directly using the ``DataBase`` object.

For start, we can load the feature information:

.. code-block:: python
   :linenos:

    >>> from actsnclass import DataBase

    >>> path_to_features_file = 'results/Bazin.dat'

    >>> data = DataBase()
    >>> data.load_features(path_to_features_file, method='Bazin')

    Loaded  21284  samples!

Notice that this data has some pre-determine separation between training and test sample:

.. code-block:: python
   :linenos:

   >>> data.metadata['sample'].unique()

   array(['test', 'train'], dtype=object)

You can choose to start your first iteration of the active learning loop from the original training sample
flagged int he file OR from scratch. As this is our first example, let's do the simple thing and start from the original
training sample. The code below build the respective samples and performs the classification:

.. code-block:: python
   :linenos:

   >>> data.build_samples(initial_training='original', nclass=2)

   Training set size:  1093
   Test set size:  20191

   >>> data.classify(method='RandomForest')
   >>> data.classprob                        # check classification probabilities

   array([[0.461, 0.539],
          [0.346, 0.654],
          ...,
          [0.398, 0.602],
          [0.396, 0.604]])

For a binary classification, the  output from the classifier for each object (line) is presented as a pair of floats, the first column
corresponding to the probability of the given object being a Ia and the second column its complement.

Given the output from the classifier we can calculate the metric(s) of choice:

.. code-block:: python
   :linenos:

   >>> data.evaluate_classification(metric_label='snpcc')
   >>> print(data.metrics_list_names)           # check metric header
    ['acc', 'eff', 'pur', 'fom']

    >>> print(data.metrics_list_values)          # check metric values
    [0.5975434599574068, 0.9024767801857585,
    0.34684684684684686, 0.13572404702012383]

and save results to file:

.. code-block:: python
   :linenos:

    >>> path_to_features_file = 'results/Bazin.dat'
    >>> metrics_file = 'results/metrics.dat'
    >>> queried_sample_file = 'results/queried_sample.da

   >>> data.save_metrics(loop=0, output_metrics_file=metrics_file)
   >>> data.save_queried_sample(loop=0, queried_sample_file=query_file,
   >>>                          full_sample=False)