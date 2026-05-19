#!/usr/bin/env python3
"""
Actor Network Link Prediction - Production Script
This script trains the model and generates predictions for submission.
"""

import numpy as np
import pandas as pd
import csv
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
SEED = 42

def load_data(train_path, test_path, node_info_path):
    """Load training, test, and node feature data."""
    
    # Load training set
    with open(train_path, "r") as f:
        reader = csv.reader(f)
        train_set = list(reader)
    train_set = [element[0].split(" ") for element in train_set]
    train_df = pd.DataFrame(train_set, columns=['source', 'target', 'label'])
    train_df['source'] = train_df['source'].astype(int)
    train_df['target'] = train_df['target'].astype(int)
    train_df['label'] = train_df['label'].astype(int)
    
    # Load test set
    with open(test_path, "r") as f:
        reader = csv.reader(f)
        test_set = list(reader)
    test_set = [element[0].split(" ") for element in test_set]
    test_df = pd.DataFrame(test_set, columns=['source', 'target'])
    test_df['source'] = test_df['source'].astype(int)
    test_df['target'] = test_df['target'].astype(int)
    
    # Load node features
    node_features = pd.read_csv(node_info_path, header=None)
    
    return train_df, test_df, node_features

def build_network(train_df, node_features):
    """Build network graph from positive training edges."""
    
    positive_edges = train_df[train_df['label'] == 1][['source', 'target']].values
    
    G = nx.Graph()
    G.add_nodes_from(range(node_features.shape[0]))
    G.add_edges_from(positive_edges)
    
    return G

def compute_graph_features(G, source, target):
    """Compute graph-theoretical features for a node pair."""
    
    features = {}
    
    # Common neighbors
    if source in G and target in G:
        common_neighbors = len(list(nx.common_neighbors(G, source, target)))
    else:
        common_neighbors = 0
    features['common_neighbors'] = common_neighbors
    
    # Jaccard similarity
    if source in G and target in G:
        neighbors_source = set(G.neighbors(source))
        neighbors_target = set(G.neighbors(target))
        union = len(neighbors_source | neighbors_target)
        features['jaccard'] = len(neighbors_source & neighbors_target) / union if union > 0 else 0
    else:
        features['jaccard'] = 0
    
    # Adamic-Adar Index
    if source in G and target in G:
        adamic_adar = sum([1.0/np.log(G.degree(z)) 
                          for z in nx.common_neighbors(G, source, target) 
                          if G.degree(z) > 1])
    else:
        adamic_adar = 0
    features['adamic_adar'] = adamic_adar
    
    # Preferential attachment
    source_degree = G.degree(source) if source in G else 0
    target_degree = G.degree(target) if target in G else 0
    features['pref_attachment'] = source_degree * target_degree
    features['degree_source'] = source_degree
    features['degree_target'] = target_degree
    
    # Shortest path
    if source in G and target in G:
        try:
            shortest_path = nx.shortest_path_length(G, source, target)
        except nx.NetworkXNoPath:
            shortest_path = np.inf
    else:
        shortest_path = np.inf
    features['shortest_path'] = shortest_path if shortest_path != np.inf else 999
    
    return features

def compute_text_features(source_features, target_features):
    """Compute text similarity features between node pairs."""
    
    features = {}
    
    # Cosine similarity
    source_norm = np.linalg.norm(source_features)
    target_norm = np.linalg.norm(target_features)
    if source_norm > 0 and target_norm > 0:
        cosine_sim = np.dot(source_features, target_features) / (source_norm * target_norm)
    else:
        cosine_sim = 0
    features['cosine_similarity'] = cosine_sim
    
    # L2 distance
    features['l2_distance'] = np.linalg.norm(source_features - target_features)
    
    # Dot product
    features['dot_product'] = np.dot(source_features, target_features)
    
    # Shared features
    source_nonzero = (source_features > 0).astype(int)
    target_nonzero = (target_features > 0).astype(int)
    shared_features = np.sum(source_nonzero * target_nonzero)
    features['shared_features'] = shared_features
    
    # L1 distance
    features['l1_distance'] = np.sum(np.abs(source_features - target_features))
    
    return features

def extract_features(df, G, node_feature_matrix, is_training=True):
    """Extract all features for a set of node pairs."""
    
    print(f"Extracting features for {len(df)} pairs...")
    
    graph_features = []
    text_features = []
    
    for idx, row in df.iterrows():
        # Graph features
        gf = compute_graph_features(G, row['source'], row['target'])
        graph_features.append(gf)
        
        # Text features
        source_features = node_feature_matrix[row['source']]
        target_features = node_feature_matrix[row['target']]
        tf = compute_text_features(source_features, target_features)
        text_features.append(tf)
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1} pairs")
    
    graph_df = pd.DataFrame(graph_features)
    text_df = pd.DataFrame(text_features)
    
    # Combine features
    X = pd.concat([graph_df, text_df], axis=1)
    
    # Clean data
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 999)
    
    return X

def train_and_predict(train_path, test_path, node_info_path, output_path='predictions.csv'):
    """Main function: train model and generate predictions."""
    
    print("=" * 60)
    print("Actor Network Link Prediction - Training Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    train_df, test_df, node_features = load_data(train_path, test_path, node_info_path)
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Node features: {node_features.shape}")
    
    # Build network
    print("\n[2] Building network graph...")
    G = build_network(train_df, node_features)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")
    
    # Extract node feature matrix
    node_feature_matrix = node_features.iloc[:, 1:].values
    
    # Extract features for training set
    print("\n[3] Extracting features for training set...")
    X_train = extract_features(train_df, G, node_feature_matrix, is_training=True)
    y_train = train_df['label'].values
    print(f"  Training features shape: {X_train.shape}")
    
    # Extract features for test set
    print("\n[4] Extracting features for test set...")
    X_test = extract_features(test_df, G, node_feature_matrix, is_training=False)
    print(f"  Test features shape: {X_test.shape}")
    
    # Scale features
    print("\n[5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Features scaled")
    
    # Train model
    print("\n[6] Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        subsample=0.8,
        random_state=SEED
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Make predictions
    print("\n[7] Generating predictions...")
    test_predictions = model.predict_proba(X_test_scaled)[:, 1]
    print(f"  Predictions shape: {test_predictions.shape}")
    print(f"  Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
    
    # Create submission
    print("\n[8] Creating submission file...")
    submission = pd.DataFrame({
        'ID': range(len(test_predictions)),
        'Predictions': test_predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"  Submission saved to '{output_path}'")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return model, scaler, X_train.columns

if __name__ == "__main__":
    # Run the pipeline
    train_and_predict(
        train_path='train.txt',
        test_path='test.txt',
        node_info_path='node_information.csv',
        output_path='predictions.csv'
    )
