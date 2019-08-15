.. _learnloop:


Active Learning loop
====================

Details on running 1 loop
-------------------------

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

.. hint:: If you wish to start from scratch, just set the `initial_training=N` where `N` is the number of objects in you want in the initial training. The code will then randomly select `N` objects from the entire sample as the initial training sample. It will also impose that at least half of them are SNe Ias.

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

and save results for this one loop to file:

.. code-block:: python
   :linenos:

    >>> path_to_features_file = 'results/Bazin.dat'
    >>> metrics_file = 'results/metrics.dat'
    >>> queried_sample_file = 'results/queried_sample.dat'

   >>> data.save_metrics(loop=0, output_metrics_file=metrics_file)
   >>> data.save_queried_sample(loop=0, queried_sample_file=query_file,
   >>>                          full_sample=False)

You should now have in your ``results`` directory a ``metrics.dat`` file which looks like this:

.. literalinclude:: images/example_diagnostic.dat
 :lines: 1-2


Running a number of iterations in sequence
------------------------------------------

We provide a function where all the above steps can be done in sequence for a number of iterations.
In interactive mode, you must define the required variables and use the :py:mod:`actsnclass.learn_loop` function:

.. code-block:: python
   :linenos:

   >>> from actsnclass import  learn_loop

   >>> nloops = 1000                                  # number of iterations
   >>> method = 'Bazin'                               # only option in v1.0
   >>> ml = 'RandomForest'                            # only option in v1.0
   >>> strategy = 'RandomSampling'                    # learning strategy
   >>> input_file = 'results/Bazin.dat'               # input features file
   >>> diag = 'results/diagnostic.dat'                # output diagnostic file
   >>> queried = 'results/queried.dat'                # output query file
   >>> train = 'original'                             # initial training
   >>> batch = 1                                      # size of batch

   >>> learn_loop(nloops=nloops, features_method=method, classifier=ml,
   >>>            strategy=strategy, path_to_features=input_file, output_diag_file=diag,
   >>>            output_queried_file=queried, training=train, batch=batch)

Alternatively you can also run everything from the command line:

.. code-block:: bash

   >>> run_loop.py -i <input features file> -b <batch size> -n <number of loops>
   >>>             -d <output metrics file> -q <output queried sample file>
   >>>             -s <learning strategy> -t <choice of initial training>

The queryable sample
--------------------

In the example shown above, when reading the data from the features file there was only 2 possibilities for the
`sample` variable:

.. code-block:: python
   :linenos:

   >>> data.metadata['sample'].unique()
   array(['test', 'train'], dtype=object)

This corresponds to an unrealistic scenario where we are able to obtain spectra for any object at any time.

.. hint:: If you wish to restrict the sample available for querying, just change the `sample` variable to `queryable` for the objects available for querying. Whenever this keywork is encountered in a file of extracted features, the code automatically restricts the query selection to the objects flagged as `queryable`.


Active Learning loop in time domain
===================================

Considering that you have previously prepared the time domain data, you can run the active learning loop
in its current form either by using the :py:mod:`actsnclass.time_domain_loop` or by using the command line
interface:

.. code-block:: bash

    >>> run_time_domain.py -d <first day of survey> <last day of survey>
    >>>        -m <output metrics file> -q <output queried file> -f <features directory>
    >>>        -s <learning strategy> -fm <path to full light curve features file >