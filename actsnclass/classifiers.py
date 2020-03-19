# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 10 August 2019
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['random_forest','gradient_boosted_trees','knn',
           'mlp','svm','nbg', 'bootstrap_clf']

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample


def bootstrap_clf(clf_function, n_ensembles, train_features,
                  train_labels, test_features, **kwargs):
    n_labels = np.unique(train_labels).size
    num_test_data = test_features.shape[0]
    ensemble_probs = np.zeros((num_test_data, n_ensembles, n_labels))
    for i in range(n_ensembles):
        x_train, y_train = resample(train_features, train_labels)
        predicted_class, class_prob = clf_function(x_train,
                                                   y_train,
                                                   test_features,
                                                   **kwargs)
        ensemble_probs[:, i, :] = class_prob

    class_prob = ensemble_probs.mean(axis=1)
    predictions = np.argmax(class_prob, axis=1)
    return predictions, class_prob, ensemble_probs


def random_forest(train_features:  np.array, train_labels: np.array,
                  test_features: np.array, **kwargs):
    """Random Forest classifier.

    Parameters
    ----------
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All keywords required by
        sklearn.ensemble.RandomForestClassifier function.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    # create classifier instance
    clf = RandomForestClassifier(**kwargs)
    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob


def gradient_boosted_trees(train_features: np.array,
                           train_labels: np.array,
                           test_features: np.array, **kwargs):
    """Gradient Boosted Trees classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.XGBClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = XGBClassifier(**kwargs)

    clf.fit(train_features, train_labels)             # train
    predictions = clf.predict(test_features)          # predict
    prob = clf.predict_proba(test_features)           # get probabilities

    return predictions, prob


def knn(train_features: np.array, train_labels: np.array,
        test_features: np.array, **kwargs):

    """K-Nearest Neighbour classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.neighbors.KNeighborsClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = KNeighborsClassifier(**kwargs)

    clf.fit(train_features, train_labels)              # train
    predictions = clf.predict(test_features)           # predict
    prob = clf.predict_proba(test_features)            # get probabilities

    return predictions, prob

def mlp(train_features: np.array, train_labels: np.array,
        test_features: np.array, **kwargs):

    """Multi Layer Perceptron classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: extra parameters
        All parameters allowed by sklearn.neural_network.MLPClassifier

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf=MLPClassifier(**kwargs)

    clf.fit(train_features, train_labels)              # train
    predictions = clf.predict(test_features)           # predict
    prob = clf.predict_proba(test_features)            # get probabilities

    return predictions, prob

def svm(train_features: np.array, train_labels: np.array,
        test_features: np.array, **kwargs):
    """Support Vector classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: dict (optional)
        All parameters which can be passed to sklearn.svm.SVC
        function.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = SVC(kwargs)

    clf.fit(train_features, train_labels)          # train
    predictions = clf.predict(test_features)       # predict
    prob = clf.predict_proba(test_features)        # get probabilities

    return predictions, prob

def nbg(train_features: np.array, train_labels: np.array,
                  test_features: np.array, **kwargs):

    """Naive Bayes classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    kwargs: dict (optional)
        All parameters which can be passed to sklearn.svm.SVC
        function.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf=GaussianNB(**kwargs)

    clf.fit(train_features, train_labels)         # fit
    predictions = clf.predict(test_features)      # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob


def main():
    return None


if __name__ == '__main__':
    main()
