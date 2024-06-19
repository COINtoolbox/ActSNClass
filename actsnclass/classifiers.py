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

__all__ = ['mlflow_tracking_And_Registry','random_forest']

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime

def mlflow_tracking_And_Registry(clf, train_features):
    # Set the MLflow experiment
    mlflow.set_experiment("Random_Forest_Experiment")
    
    # Enable autolog
    mlflow.sklearn.autolog()

    train_size = train_features.shape[0]

    # Generate the run name with the train size
    run_name = f"Train_size_{train_size}"

    # Get the current date in day-month-year format
    current_date = datetime.now().strftime("%d-%m-%Y")

    # Generate the model name with the current date
    model_name = f"Random_Forest_Experiment_Vi_{current_date}"

    # Start a new run
    with mlflow.start_run(run_name=run_name) as run:
        # Log the model with the generated name
        mlflow.sklearn.log_model(clf, model_name)


        # Register the model in the Model Registry
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
        registered_model = mlflow.register_model(model_uri, model_name)

        # Add a description or change the stage of the model version if needed
        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=registered_model.version,
            description="Random Forest model registered on " + current_date
        )


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
#    mlflow.sklearn.autolog()
    # Ensure no previous autolog is active
    mlflow.autolog(disable=True)
    
    # Enable autolog before model training
    mlflow.sklearn.autolog()
    # create classifier instance
    clf = RandomForestClassifier(n_estimators=nest, random_state=seed,
                                 max_depth=max_depth, n_jobs=n_jobs)
    clf.fit(train_features, train_labels)                     # train
    predictions = clf.predict(test_features)                # predict
    prob = clf.predict_proba(test_features)       # get probabilities
    


    
    # Call mlflow_tracking_And_Registry function to handle MLflow logging
    mlflow_tracking_And_Registry(clf, train_features)

    return predictions, prob


def main():
    return None


if __name__ == '__main__':
    main()
