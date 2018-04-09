"""Created by the CRP #4 team between 20-27 Aug 2017.

Functions to deal with the naming conventions on post-SNPCC data.

- snid2filename
    Convert snid to complete raw data file path.
"""

import os

def snid2filename(snid,data_path='.'): 
    """
    Convert snid to complete raw data file path.

    input: snid, str or int - supernova identification
           data_path, str - path to raw data folder

    output: str, complete path to raw data file.
    """

    # sanatize path
    sn_file = "DES_SN%s.DAT" % str(snid).zfill(6)

    return os.path.join(data_path,sn_file)
