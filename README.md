# Machine Learning A→Z — From Scratch to Production

A practical, beginner‑friendly roadmap to learn ML **step by step** and apply models to **CSV/Excel/JSON/Parquet/SQL**, plus unsupervised learning, tuning, feature engineering, deep learning, deployment, and responsible AI.

---

## 📌 What you’ll build

* Reproducible Python environment
* End‑to‑end ML **pipelines** (preprocess → train → evaluate → explain)
* Recipes to train on **CSV and other file types** (Excel, JSON, Parquet, SQL)
* Unsupervised clustering + dimensionality reduction
* Model **tuning** (CV, Grid/Random search)
* Feature engineering & interpretability (Permutation Importance, SHAP)
* A mini **FastAPI** service (deployable in Docker)
* **Fairness** audit with basic mitigation

---

## 🧭 Table of Contents

1. [Quickstart (5 minutes)](#quickstart-5-minutes)
2. [Environment Setup](#environment-setup)
3. [Repo Structure](#repo-structure)
4. [Loading Data: CSV, Excel, JSON, Parquet, SQL, Images, Text](#loading-data)
5. [Supervised Learning: Universal Pipeline (CSV & friends)](#supervised-learning-universal-pipeline)
6. [Unsupervised Learning: k‑Means + PCA](#unsupervised-learning)
7. [Evaluation & Hyper‑parameter Tuning](#evaluation--hyper-parameter-tuning)
8. [Feature Engineering & Interpretability](#feature-engineering--interpretability)
9. [Deep Learning Starter (PyTorch *or* Keras)](#deep-learning-starter)
10. [Deploy with FastAPI (+ Docker)](#deploy-with-fastapi--docker)
11. [Responsible AI: Fairness Audit](#responsible-ai-fairness-audit)
12. [Makefile / Task Runner (optional)](#makefile--task-runner-optional)
13. [FAQ](#faq)

---

## Quickstart (5 minutes)

```bash
# 1) Create and activate a clean environment (conda or venv)
conda create -n ml101 python=3.11 -y && conda activate ml101
#    or: python -m venv .venv && source .venv/bin/activate  # (Windows: .\.venv\Scripts\activate)

# 2) Install core packages
pip install numpy pandas scikit-learn jupyterlab matplotlib seaborn xgboost shap lime fastapi uvicorn[standard] joblib category-encoders

# 3) Launch JupyterLab
jupyter lab
```

Open `notebooks/01_quickstart.ipynb` (create it if it doesn’t exist) and paste the **Universal Pipeline** from below.

---

## Environment Setup

**Option A: Conda**

```bash
conda update -n base -c defaults conda
conda create -n ml101 python=3.11 numpy pandas scikit-learn jupyterlab matplotlib seaborn -y
conda activate ml101
pip install xgboost shap lime fastapi uvicorn[standard] joblib category-encoders
```

**Option B: venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .\.venv\Scripts\activate
pip install -U pip
pip install numpy pandas scikit-learn jupyterlab matplotlib seaborn xgboost shap lime fastapi uvicorn[standard] joblib category-encoders
```

Create a minimal **requirements.txt** (optional for deployment):

```text
numpy
pandas
scikit-learn
xgboost
category-encoders
matplotlib
seaborn
shap
lime
fastapi
uvicorn[standard]
joblib
```

---

## Repo Structure

```
ml-a2z/
├─ data/                 # raw & processed data (gitignore large files)
├─ notebooks/            # exploration & mini-projects
├─ src/                  # reusable code
│  ├─ data_loading.py
│  ├─ pipelines.py
│  ├─ train.py
│  └─ serve_fastapi.py
├─ models/               # saved models (joblib / onnx)
├─ reports/              # metrics, drift/fairness reports
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## Loading Data

### Tabular files

```python
import pandas as pd
# CSV
df = pd.read_csv("data/your_file.csv")
# Excel
# df = pd.read_excel("data/your_file.xlsx", sheet_name=0)
# JSON (records or table orient work well)
# df = pd.read_json("data/your_file.json")
# Parquet (fast, typed)
# df = pd.read_parquet("data/your_file.parquet")
```

### SQL

```python
import pandas as pd, sqlite3
conn = sqlite3.connect("data/example.db")
df = pd.read_sql_query("SELECT * FROM table_name", conn)
```

### Images (directory of class folders)

```python
# PyTorch
from torchvision import datasets, transforms
train_ds = datasets.ImageFolder(
    root="data/images/train",
    transform=transforms.ToTensor()
)
```

### Text (Hugging Face datasets)

```python
from datasets import load_dataset
imdb = load_dataset("imdb")
```

---

## Supervised Learning: Universal Pipeline

Use this for **classification** or **regression** with mixed numeric/categorical features. Swap in any estimator.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from category_encoders.target_encoder import TargetEncoder
import numpy as np

# 1) Load your data
# df = pd.read_csv("data/your_file.csv")

# Example split
TARGET = "target"  # change me
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()<=20 else None
)

# 2) Preprocess: numeric scale + categorical encode
num_cols = X.select_dtypes(include=["number"]).columns
cat_cols = X.select_dtypes(exclude=["number"]).columns

numeric_pipe = Pipeline([
    ("scale", StandardScaler())
])

# choose one: OneHot (safe) OR TargetEncoder (great for high-cardinality)
categorical_pipe = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
# categorical_pipe = Pipeline([("target", TargetEncoder())])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

# 3) Pick a model (classification or regression)
if y.nunique() <= 20 and set(y.unique()) <= {0,1}:  # binary classification heuristic
    model = RandomForestClassifier(n_estimators=300, random_state=42)
else:
    # choose according to task
    model = RandomForestRegressor(n_estimators=300, random_state=42)

pipe = Pipeline([("prep", preprocess), ("model", model)])
pipe.fit(X_train, y_train)

# 4) Evaluate
if y.nunique() <= 20 and set(y.unique()) <= {0,1}:
    y_pred = pipe.predict(X_val)
    print(classification_report(y_val, y_pred))
else:
    rmse = mean_squared_error(y_val, pipe.predict(X_val), squared=False)
    print("RMSE:", round(rmse, 3))

# 5) Save for deployment
import joblib
joblib.dump(pipe, "models/model.joblib")
```

**Swap-in models** (one line change):

* `from sklearn.linear_model import LogisticRegression; model = LogisticRegression(max_iter=1000)`
* `from xgboost import XGBClassifier; model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.7, colsample_bytree=0.8, random_state=42)`
* `from sklearn.neighbors import KNeighborsClassifier; model = KNeighborsClassifier(n_neighbors=15)` (remember to scale numerics)

---

## Unsupervised Learning

k‑Means + PCA quick recipe on numeric columns:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

X = df.select_dtypes(include="number").copy()
X_std = StandardScaler().fit_transform(X)

scores = {}
for k in range(2, 11):
    labels = KMeans(n_clusters=k, n_init="auto", random_state=0).fit_predict(X_std)
    scores[k] = silhouette_score(X_std, labels)

best_k = max(scores, key=scores.get)
km = KMeans(n_clusters=best_k, n_init="auto", random_state=0)
labels = km.fit_predict(X_std)

df["cluster"] = labels

pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_std)

import matplotlib.pyplot as plt
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, s=20)
plt.title(f"k-Means (k={best_k}) on PCA")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.show()
```

---

## Evaluation & Hyper‑parameter Tuning

```python
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

param_grid = {
  "model__n_estimators": [100, 300, 500],
  "model__max_depth": [None, 8, 16],
  "model__max_features": ["sqrt", 0.5, 0.8]
}

gcv = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1)
gcv.fit(X_train, y_train)
print("Best params:", gcv.best_params_)
print("CV best score:", round(gcv.best_score_, 3))

best_model = gcv.best_estimator_
```

---

## Feature Engineering & Interpretability

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Example: extract year, month, dow from a datetime column
X["date"] = pd.to_datetime(X["date"])  # ensure datetime dtype

date_feat = FunctionTransformer(
    lambda s: np.c_[s.dt.year, s.dt.month, s.dt.dayofweek],
    feature_names_out=lambda _, f: ["year", "month", "dow"]
)
```

**Permutation Importance** (model‑agnostic):

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(best_model, X_val, y_val, scoring="f1", n_repeats=20)
importances = pd.Series(result.importances_mean, index=best_model["prep"].get_feature_names_out())
print(importances.sort_values(ascending=False).head(10))
```

**SHAP** (tree models are fast):

```python
import shap
explainer = shap.TreeExplainer(best_model["model"])  # for RandomForest/XGBoost
Xv = best_model["prep"].transform(X_val)
shap_values = explainer.shap_values(Xv)
shap.summary_plot(shap_values, feature_names=best_model["prep"].get_feature_names_out())
```

---

## Deep Learning Starter

### PyTorch (MNIST MLP)

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Flatten(), nn.Linear(28*28,256), nn.ReLU(),
            nn.Linear(256,64), nn.ReLU(), nn.Linear(64,10)
        )
    def forward(self, x): return self.seq(x)

model = Net().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(); pred = model(x); loss = loss_fn(pred,y)
        loss.backward(); opt.step()
    print(f"epoch {epoch+1} loss {loss.item():.4f}")
```

### Keras (MNIST MLP)

```python
import tensorflow as tf
from tensorflow.keras import layers
(xtr,ytr),(xv,yv) = tf.keras.datasets.mnist.load_data()
xtr,xv = xtr/255.0, xv/255.0
model = tf.keras.Sequential([layers.Flatten(), layers.Dense(256,activation='relu'), layers.Dense(64,activation='relu'), layers.Dense(10,activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(xtr,ytr, validation_data=(xv,yv), epochs=3, batch_size=128)
```

---

## Deploy with FastAPI & Docker

**app.py**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

class InputRow(BaseModel):
    # EDIT to match your schema
    pclass:int; sex:str; age:float; fare:float

app = FastAPI(title="ML Model API")
model = joblib.load("models/model.joblib")

@app.post("/predict")
def predict(row: InputRow):
    X = pd.DataFrame([row.dict()])
    proba = getattr(model, 'predict_proba', model.predict)(X)
    # handle proba for classifiers vs prediction for regressors
    try:
        proba = float(proba[0,1])
        return {"positive_class_probability": round(proba, 3)}
    except Exception:
        return {"prediction": float(proba[0])}
```

**Run locally**

```bash
pip install fastapi uvicorn[standard] pandas joblib
uvicorn app:app --reload
# Open http://127.0.0.1:8000/docs
```

**Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT=80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
```

---

## Responsible AI: Fairness Audit

```python
pip install fairlearn

from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv("data/adult.csv")
y = (df["income"] == ">50K").astype(int)
X = df.drop(columns=["income"])

sens = df["sex"]
X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(X, y, sens, stratify=y, test_size=0.2, random_state=0)

pre = ColumnTransformer([
    ("num", StandardScaler(), X.select_dtypes("number").columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), X.select_dtypes("object").columns)
])

base = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=2000))])
base.fit(X_tr, y_tr)

pred = base.predict(X_te)
mf = MetricFrame(metrics={"acc": accuracy_score, "dp": demographic_parity_difference, "eo": equalized_odds_difference}, y_true=y_te, y_pred=pred, sensitive_features=s_te)
print("Baseline fairness metrics:\n", mf.by_group)

mit = ExponentiatedGradient(base, constraints=DemographicParity())
mit.fit(X_tr, y_tr, sensitive_features=s_tr)
pred_m = mit.predict(X_te)
mf_m = MetricFrame(metrics={"acc": accuracy_score, "dp": demographic_parity_difference, "eo": equalized_odds_difference}, y_true=y_te, y_pred=pred_m, sensitive_features=s_te)
print("Mitigated fairness metrics:\n", mf_m.by_group)
```

Add a **Model Card** (markdown) describing data, metrics, fairness, limitations, and contact for redress.

---

## Makefile / Task Runner (optional)

```makefile
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

lab:
	jupyter lab

train:
	python -m src.train --data data/your_file.csv --target target

serve:
	uvicorn src.serve_fastapi:app --reload

docker:
	docker build -t ml-a2z:latest .
```

---

## FAQ

**Q: Which model should I start with?**  Logistic/Linear Regression for baselines; Random Forest/XGBoost for strong tabular performance.

**Q: How do I handle missing values?**  Use `SimpleImputer` inside your `ColumnTransformer`; never impute on the full dataset before splitting.

**Q: My classes are imbalanced.**  Use `class_weight='balanced'`, stratified CV, and monitor Precision‑Recall AUC.

**Q: How do I switch to regression?**  Swap the estimator to `RandomForestRegressor`/`XGBRegressor` and metrics to RMSE/MAE.

**Q: Where should big data live?**  Keep large files outside git; store paths in a `.env` or config file.

---

> ✨ Tip: Duplicate this README as a template for each new project. Replace schema, target name, and paths; the **pipelines** stay the same.
