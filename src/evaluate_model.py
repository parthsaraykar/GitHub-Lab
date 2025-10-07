import os
import pickle
import mlflow
import json
import random
import datetime
from joblib import load
from sklearn.datasets import fetch_rcv1
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import argparse

# --------------------------------------------
# Parse Arguments (run_id or timestamp)
# --------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=False, help="Run ID from GitHub Actions")
args = parser.parse_args()
run_id = args.run_id if args.run_id else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --------------------------------------------
# Load Data (use cached pickles or fetch)
# --------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
METRICS_DIR = "metrics"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

if os.path.exists(os.path.join(DATA_DIR, "data.pickle")) and os.path.exists(os.path.join(DATA_DIR, "target.pickle")):
    print("📂 Loading cached RCV1 dataset for evaluation...")
    X = pickle.load(open(os.path.join(DATA_DIR, "data.pickle"), "rb"))
    y = pickle.load(open(os.path.join(DATA_DIR, "target.pickle"), "rb"))
else:
    print("⬇️ Fetching Reuters Corpus Volume (RCV1) dataset...")
    rcv1 = fetch_rcv1(subset="test")
    X = rcv1.data
    y = rcv1.target
    pickle.dump(X, open(os.path.join(DATA_DIR, "data.pickle"), "wb"))
    pickle.dump(y, open(os.path.join(DATA_DIR, "target.pickle"), "wb"))

# --------------------------------------------
# Handle label dimensionality safely
# --------------------------------------------
if sp.issparse(y):
    y = y.toarray()

if len(y.shape) > 1:
    # Multi-label dataset (RCV1)
    y = y[:, random.randint(0, y.shape[1] - 1)]
else:
    # Single-label dataset
    y = y

# --------------------------------------------
# Locate the Most Recent Model
# --------------------------------------------
model_files = sorted(
    [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")],
    key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)),
    reverse=True,
)
if not model_files:
    raise FileNotFoundError(" No model found in models/ directory!")

model_path = os.path.join(MODEL_DIR, model_files[0])
print(f"🧠 Evaluating model: {model_path}")

# --------------------------------------------
# Evaluate
# --------------------------------------------
model = load(model_path)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
y_pred = model.predict(test_X)

acc = accuracy_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
print(f" Accuracy: {acc:.3f} | F1 Score: {f1:.3f}")

# --------------------------------------------
# Log Metrics and Save Locally
# --------------------------------------------
mlflow.set_tracking_uri("./mlruns")
with mlflow.start_run(run_name=f"Eval_{run_id}"):
    mlflow.log_metrics({"eval_accuracy": acc, "eval_f1": f1})

metrics = {
    "run_id": run_id,
    "model_file": model_path,
    "accuracy": round(acc, 3),
    "f1_score": round(f1, 3),
    "timestamp": datetime.datetime.now().isoformat(),
}
metrics_file = os.path.join(METRICS_DIR, f"{run_id}_metrics.json")

with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

print(f" Metrics saved to {metrics_file}")
