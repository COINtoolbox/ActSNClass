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

__all__ = ['random_forest']

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def random_forest(train_features:  np.array, train_labels: np.array,
                  test_features: np.array, nest=1000, seed=42, max_depth=None,
                  n_jobs=1):
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
    max_depth: None or int (optional)
        The maximum depth of the tree. Default is None.
    n_jobs: int (optional)
            Number of cores used to train the model. Default is 1.
        

    Returns
    -------
    predictions: np.array
        Predicted classes.
    prob: np.array
        Classification probability for all objects, [pnon-Ia, pIa].
    """

    # create classifier instance
    '''    clf = RandomForestClassifier(n_estimators=nest, random_state=seed,
                                 max_depth=max_depth, n_jobs=n_jobs)
    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities
    '''

    # Initialiser MLflow
    mlflow.start_run()

    try:
        # Enregistrer les hyperparamètres
        mlflow.log_param("n_estimators", nest)
        mlflow.log_param("random_state", seed)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_jobs", n_jobs)

        # Créer une instance du classificateur
        clf = RandomForestClassifier(n_estimators=nest, random_state=seed,
                                     max_depth=max_depth, n_jobs=n_jobs)
        clf.fit(train_features, train_labels)  # Entraîner le modèle

        # Prédire
        predictions = clf.predict(test_features)
        prob = clf.predict_proba(test_features)  # Obtenir les probabilités

        
        # Enregistrer le modèle
        mlflow.sklearn.log_model(clf, "random_forest_model")

    finally:
        mlflow.end_run()

    return predictions, prob


def main():
    return None


if __name__ == '__main__':
    main()
