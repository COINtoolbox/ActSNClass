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

__all__ = ['random_forest','gradient_boosted_trees','knn_classifier','mlp_classifier','svm_classifier','nbg_classifier']

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def random_forest(train_features:  np.array, train_labels: np.array,
                  test_features: np.array, nest=1000, seed=42):
    """Random Forest classifier.

    Parameters
    ----------
    train_features: np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    nest: int (optional)
        Number of estimators (trees) in the forest.
        Default is 1000.
    seed: float (optional)
        Seed for random number generator. Default is 42.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    # create classifier instance
    clf = RandomForestClassifier(n_estimators=nest, 
                                 random_state=seed,max_depth=None,
                                 n_jobs=2,bootstrap=True,
                                 criterion="gini",
                                 verbose=0, oob_score=True)
    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities

    return predictions, prob


def gradient_boosted_trees(train_features: np.array, 
                           train_labels: np.array,
                           test_features: np.array, nest = 1000,
                            seed = 42):


    """Gradient Boosted Trees classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    nest: int (optional)
        Number of estimators (trees) in the forest.
        Default is 1000.
    seed: float (optional)
        Seed for random number generator. Default is 42.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = XGBClassifier(max_depth=6,learning_rate=0.05,
                        n_estimators=nest,silent=True,
                        objective="binary:logistic",
                        nthread=4,random_state=seed,n_folds=5,
                        min_child_weight=0.5)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    prob = clf.predict_proba(test_features)  

    return predictions, prob


def knn_classifier(train_features: np.array, train_labels: np.array,
                   test_features: np.array, nnbrs = 50, seed = 42):

    """K-Nearest Neighbour classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    nnbrs : int (optional)
        Number of neighbours. Default is 50.
    seed: float (optional)
        Seed for random number generator. Default is 42.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """

    #create classifier instance
    clf = KNeighborsClassifier(n_neighbors=nnbrs,n_jobs=-1,weights='distance',p=2)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    prob = clf.predict_proba(test_features)  

    return predictions, prob

def mlp_classifier(train_features: np.array, train_labels: np.array,
                   test_features: np.array, n_neur = 100, seed = 42):

    """Multi Layer Perceptron classifier.

    Parameters
    ----------
    train_features : np.array
        Training sample features.
    train_labels: np.array
        Training sample classes.
    test_features: np.array
        Test sample features.
    n_neur : int (optional)
        Number of neurons in the hidden layer. Default is 100.
    seed: float (optional)
        Seed for random number generator. Default is 42.

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    """
    
    #create classifier instance
    clf=MLPClassifier(hidden_layer_sizes=(n_neur), max_iter=250, 
                      alpha=1e-4,solver='adam', verbose=False, tol=1e-4,
                      random_state=seed,learning_rate_init=.001,early_stopping=True)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    prob = clf.predict_proba(test_features)
        
    return predictions, prob

def svm_classifier(train_features: np.array, train_labels: np.array,
                   test_features: np.array, seed = 42):
    clf = SVC(kernel='rbf', gamma = 'scale',probability = True,class_weight='balanced')
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    prob = clf.predict_proba(test_features)

    return predictions, prob

def nbg_classifier(train_features: np.array, train_labels: np.array,
                  test_features: np.array, seed = 42):
    clf=GaussianNB()
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    prob = clf.predict_proba(test_features)
    
    return predictions, prob


def main():
    return None


if __name__ == '__main__':
    main()
