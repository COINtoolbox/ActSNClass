from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import numpy as np
import pandas as pd

client = MlflowClient()

"""# Transitionner la version 1 de "RandomForestModel" à "Staging"
client.transition_model_version_stage(
    name="Random_Forest_Experiment_18-06-2024",
    version=2,
    stage="Staging"
) """

# Charger la version 1 du modèle en stage "Staging"
model = mlflow.pyfunc.load_model(model_uri="models:/Random_Forest_Experiment_Vi_19-06-2024/5")


data = np.random.rand(100, 12)

feature_names = [f"feature_{i+1}" for i in range(12)]
df = pd.DataFrame(data, columns=feature_names)

X_test = df.values
predictions = model.predict(X_test)
print(X_test)
print(predictions)


