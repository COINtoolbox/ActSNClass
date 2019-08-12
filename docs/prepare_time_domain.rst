.. _time_domain:

Prepare data for time domain
============================

In order to mimic the realistic situation where only a limited number of observed epochs is available at each
day, it is necessary to prepare our simulate data resemble this scenario. In ``actsnclass`` this is done in
5 steps:

1. Determine minimum and maximum MJD for the entire SNPCC sample;

2. For each day of the survey, run through the entire data sample and select only the observed epochs which were obtained prior to it;

3. Perform the feature extraction process considering only the photometric points which survived item 2.

4. Check if in the MJD in question the object is available for querying.

5. Join all information in a standard features file.


You can perform the entire analysis for one day of the survey using the :py:mod:`actsnclass.time_domain` module:

.. code-block:: python
   :linenos:

   >>> from actsnclass import SNPCCPhotometry

   >>> path_to_data = 'data/SIMGEN_PUBLIC_DES/'
   >>> output_dir = 'results/time_domain/'
   >>> day = 20

   >>> data = SNPCCPhotometry()
   >>> data.create_daily_file(output_dir=output_dir, day=day)
   >>> data.build_one_epoch(raw_data_dir=path_to_data, day_of_survey=day,
                             time_domain_dir=output_dir)


Alternatively you can use the command line to prepare a sequence of days in one batch:

.. code-block:: bash

   >>> build_time_domain.py -d 20 21 22 23 -p <path to raw data dir> -o <path to output time domain dir>

