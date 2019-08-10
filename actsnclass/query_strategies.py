"""
Created by Emille Ishida on 3 August 2019.
"""

import numpy as np


def uncertainty_sampling(class_prob, test_ids, queryable_ids, batch=1):
    """Search for the sample with highest uncertainty in predicted class."""

    # calculate distance to the decision boundary - only binary classification
    dist = abs(class_prob[:, 1] - 0.5)

    # get indexes in increasing order
    order = dist.argsort()

    # only allow objects in the query sample to be chosen
    flag = []
    for item in order:
        if test_ids[item] in queryable_ids:
            flag.append(True)
        else:
            flag.append(False)

    flag = np.array(flag)

    final_order = order[flag]

    # return the index of the highest uncertain objects which are queryable
    return list(final_order[:batch])


def random_sampling(test_ids, queryable_ids, batch=1, seed=42):
    """Randomly choose an object from the test sample"""

    np.random.seed(seed)
    indx = np.random.randint(low=0, high=len(test_ids), size=len(test_ids))

    flag = []
    for item in indx:
        if test_ids[item] in queryable_ids:
            flag.append(True)

        else:
            flag.append(False)

    flag = np.array(flag)

    # only allow objects in the queryable sample to be chosen
    return list(indx[flag][:batch])
