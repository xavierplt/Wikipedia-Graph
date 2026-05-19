# Link Prediction in an Actor Co-occurrence Network

**CentraleSupélec — MLNS 2026 Kaggle Challenge**  
Team JDRX: Julian · Dani · Riccardo · Xavier

---

## Problem

Given a partially observed actor co-occurrence network, predict whether a link exists between each pair of actors in the test set. The evaluation metric is **ROC-AUC**.

Each node represents an actor; an edge between two actors means they co-appeared in the same context. The training set contains labelled pairs (1 = link exists, 0 = no link). The test set contains unlabelled pairs to score.

---

## Repository structure

```
centralesupelec-mlns-2026/
    train.txt               # labelled pairs: source target label
    test.txt                # unlabelled pairs: source target
    node_information.csv    # sparse binary feature vectors (~1000 dims per node)
    public_baseline.py      # random prediction baseline provided by the course

actor_network_link_prediction.ipynb   # full exploration notebook
train_and_predict.py                  # production script (train → predict → submit)
```

---

## Approach

### Feature engineering

Two families of features are computed for each (source, target) pair:

**Graph-theoretical features** — built from the network of positive training edges:

| Feature | Description |
|---|---|
| `common_neighbors` | Number of shared neighbors |
| `jaccard` | Jaccard similarity of neighborhood sets |
| `adamic_adar` | Adamic-Adar index (weights common neighbors by inverse log-degree) |
| `pref_attachment` | Preferential attachment: deg(u) × deg(v) |
| `degree_source` / `degree_target` | Individual node degrees |
| `shortest_path` | Topological distance (999 if no path exists) |

**Node text features** — computed from the sparse binary node vectors in `node_information.csv`:

| Feature | Description |
|---|---|
| `cosine_similarity` | Cosine similarity between feature vectors |
| `l2_distance` | Euclidean distance |
| `dot_product` | Raw dot product |
| `shared_features` | Number of dimensions where both vectors are non-zero |
| `l1_distance` | Sum of absolute differences |

All features are standardised with `StandardScaler` before training.

### Models

Three classifiers are benchmarked with 5-fold cross-validated `GridSearchCV`:

- **Random Forest** — grid over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Gradient Boosting** — grid over `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- **Logistic Regression** — grid over `C`, `solver`

The best-performing model on validation ROC-AUC is used for final predictions.

The production script (`train_and_predict.py`) uses a fixed Gradient Boosting configuration:

```
n_estimators=200, learning_rate=0.05, max_depth=5,
min_samples_split=5, subsample=0.8, random_state=42
```

---

## Usage

### Install dependencies

```bash
pip install numpy pandas networkx scikit-learn matplotlib seaborn
```

### Run the full pipeline

```bash
cd centralesupelec-mlns-2026
python ../train_and_predict.py
```

This produces `predictions.csv` ready for Kaggle submission.

### Explore and tune

Open `actor_network_link_prediction.ipynb` for the full exploratory pipeline, model comparison, feature importance plots, and ROC curves.

---

## Output format

```csv
ID,Predictions
0,0.7231
1,0.1045
...
```

`Predictions` is a probability score in [0, 1]. Kaggle evaluates with ROC-AUC, so raw probabilities are submitted rather than hard labels.
