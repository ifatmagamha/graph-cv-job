import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os

def prepare_features(G, cv_ids, embeddings):
    """
    Extracts structural + semantic + community features for each CV node.
    """
    data = []
    for i, cv_id in enumerate(cv_ids):
        # 1. Structural
        deg = G.degree(cv_id) if cv_id in G else 0
        
        # 2. Community (categorical)
        comm = G.nodes[cv_id].get('community_id', -1) if cv_id in G else -1
        
        # 3. Semantic
        emb = embeddings[i]
        
        feat = np.concatenate([[deg, comm], emb])
        data.append(feat)
        
    return np.array(data)

def run_classification_experiment(graph_path, cv_ids, embeddings, labels, experiment_name):
    G = nx.read_graphml(graph_path)
    for n, d in G.nodes(data=True): d['bipartite'] = int(d['bipartite'])
    
    X = prepare_features(G, cv_ids, embeddings)
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training classifier for {experiment_name}...")
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"{experiment_name} - Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")
    return acc, f1

if __name__ == "__main__":
    df_cv = pd.read_parquet("data/processed/cv.parquet")
    cv_ids = df_cv['cv_id'].tolist()
    cv_emb = np.load("data/embeddings/cv_embeddings.npy")
    
    le = LabelEncoder()
    # Filling missing domains for a stable target
    y = le.fit_transform(df_cv['domain'].fillna('Unknown'))
    
    # Experiment 1: Base Graph
    acc_base, f1_base = run_classification_experiment(
        "results/G_bipartite_with_communities.graphml", 
        cv_ids, cv_emb, y, "Base_Graph"
    )
    
    # Experiment 2: Enriched Graph
    acc_enr, f1_enr = run_classification_experiment(
        "results/G_enriched_refined.graphml", 
        cv_ids, cv_emb, y, "Enriched_Graph"
    )
    
    # Log Comparison
    results = pd.DataFrame({
        "Metric": ["Accuracy", "Macro-F1"],
        "Base": [acc_base, f1_base],
        "Enriched": [acc_enr, f1_enr],
        "Improvement": [acc_enr - acc_base, f1_enr - f1_base]
    })
    
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/classification_comparison.csv", index=False)
    print("\nClassification Comparison Results:")
    print(results)
