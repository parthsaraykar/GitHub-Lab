import os
import sys
import mlflow
import pickle
import random
import datetime
import argparse
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Make sure relative paths work correctly
sys.path.insert(0, os.path.abspath('..'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="Run ID passed from GitHub Actions")
    args = parser.parse_args()
    run_id = args.run_id

    print(f"🚀 Starting training job for Run ID: {run_id}")

    # -------------------------------
    # Step 1: Create synthetic dataset
    # -------------------------------
    X, y = make_classification(
        n_samples=random.randint(800, 1600),
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )

    os.makedirs("data", exist_ok=True)
    with open("data/data.pickle", "wb") as f:
        pickle.dump(X, f)
    with open("data/target.pickle", "wb") as f:
        pickle.dump(y, f)

    # -------------------------------
    # Step 2: Set MLflow experiment
    # -------------------------------
    mlflow.set_tracking_uri("./mlruns")
    experiment_name = f"DecisionTree_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name="DecisionTreeClassifier"):
        params = {"n_samples": X.shape[0], "n_features": X.shape[1], "algorithm": "DecisionTree"}
        mlflow.log_params(params)

        model = DecisionTreeClassifier(max_depth=6, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        mlflow.log_metrics({
            "Accuracy": round(accuracy_score(y, preds), 3),
            "F1_Score": round(f1_score(y, preds), 3)
        })

        # -------------------------------
        # Step 3: Save model inside models/
        # -------------------------------
        os.makedirs("models", exist_ok=True)
        model_filename = os.path.join("models", f"model_{run_id}_dt.joblib")
        dump(model, model_filename)

        print(f" Model saved successfully to {model_filename}")
