.. _plotting:

Plotting
========

Once you have the metrics results for a set of learning strategies you can plot the behaviour the
evolution of the metrics:

 - Accuracy: fraction of correct classifications;
 - Efficiency: fraction of total SN Ia correctly classified;
 - Purity: fraction of correct Ia classifications;
 - Figure of merit: efficiency x purity with a penalty factor of 3 for false positives (contamination).

The class `Canvas <https://actsnclass.readthedocs.io/en/latest/api/actsnclass.Canvas.html#actsnclass.Canvas>_` enables you do to it using:

.. code-block:: python
   :linenos:

   >>> from actsnclass.plot_results import Canvas

   >>> # define parameters
   >>> path_to_files = ['results/metrics_canonical.dat',
   >>>                  'results/metrics_random.dat',
   >>>                  'results/metrics_unc.dat']
   >>> strategies_list = ['Canonical', 'RandomSampling', 'UncSampling']
   >>> output_plot = 'plots/metrics.png'

   >>> #Initiate the Canvas object, read and plot the results for
   >>> # each metric and strategy.
   >>> cv = Canvas()
   >>> cv.load_metrics(path_to_files=path_to_files,
   >>>                    strategies_list=strategies_list)
   >>> cv.set_plot_dimensions()
   >>> cv.plot_metrics(output_plot_file=output_plot,
   >>>                    strategies_list=strategies_list)


This will generate:

.. image:: images/diag.png
   :align: center
   :height: 448 px
   :width: 640 px
   :alt: Plot metrics evolution.


Alternatively, you can use  it directly from the command line.

For example, the result above could also be obtained doing:

.. code-block:: bash

    >>> make_metrics_plots.py -m <path to canonical metrics> <path to rand sampling metrics>  <path to unc sampling metrics>
    >>>        -o <path to output plot file> -s Canonical RandomSampling UncSampling

OBS: the color pallete for this project was chosen to honor the work of `Piet Mondrian <https://en.wikipedia.org/wiki/Piet_Mondrian>`_.
