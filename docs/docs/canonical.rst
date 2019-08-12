.. _canonical:

Building the Canonical sample
=============================

According to the nomenclature used in `Ishida et al., 2019 <https://arxiv.org/pdf/1804.03765.pdf>`_, the Canonical
sample is a subset of the test sample chosen to hold the same characteristics of the training sample. It was used
to mimic the effect of continuously adding elements to the training sample under the traditional strategy.

It was constructed using the following steps:

#. From the raw light curve files, build a metadata matrix containing:
   ``[snid, sample, sntype, z, g_pkmag, r_pkmag, i_pkmag, z_pkmag, g_SNR, r_SNR, i_SNR, z_SNR]``
   where ``z`` corresponds to redshift, ``x_pkmag`` is the simulated peak magnitude and ``x_SNR``
   denotes the mean SNR, both in filter x;
#. Separate original training and test set in 3 subsets according to SN type: [Ia, Ibc, II];
#. For each object in the training sample, find its nearest neighbor within objects of the test sample of the
   same SN type and considering the photometric parameter space built in step 1.

This will allow you to construct a Canonical sample holding the same characteristics and size of the original training sample
but composed of different objects.

``actsnclass`` allows you to perform this task using the :mod:`Canonical <build_snpcc_canonical>` module:

.. code-block:: python
   :linenos:

   >>> from snactclass import build_snpcc_canonical

   >>> # define variables
   >>> data_dir = 'data/SIMGEN_PUBLIC_DES/'
   >>> output_sample_file = 'results/Bazin_SNPCC_canonical.dat'
   >>> output_metadata_file = 'results/Bazin_metadata.dat'
   >>> features_file = 'results/Bazin.dat'

   >>> sample = build_snpcc_canonical(path_to_raw_data: data_dir, path_to_features=features_file,
   >>>                               output_canonical_file=output_sample_file,
   >>>                               output_info_file=output_metadata_file,
   >>>                               compute=True, save=True)

Once the samples is constructed you can compare the distribution in ``[z, g_pkmag, r_pkmag]`` with a plot:

.. code-block:: python
   :linenos:

   >>> from actsnclass import plot_snpcc_train_canonical

   >>> plot_snpcc_train_canonical(sample, output_plot_file='plots/compare_canonical_train.png')

.. image:: images/canonical.png
   :align: center
   :height: 480 px
   :width: 640 px
   :alt: Comparison between original training and canonical samples.


In the command line, you can do all at once:

.. code-block:: bash

   >>> build_canonical.py -c True -d <path to raw data dir>
   >>>       -f <features file> -m <output file for metadata>
   >>>       -o <output file for canonical sample> -p <comparison plot file>
   >>>       -s True


