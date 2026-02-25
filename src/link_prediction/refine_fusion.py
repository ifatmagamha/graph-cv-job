import pandas as pd
import numpy as np
import networkx as nx
import faiss
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import os

def extract_structural_features(u, v, G):
    """
    Extracts structural features for a CV-Job pair.
    """
    if u not in G or v not in G:
        return {"pa": 0, "paths_3": 0, "deg_cv": 0, "deg_job": 0, "same_comm": 0}
    
    pa = G.degree(u) * G.degree(v)
    
    # Paths of length 3 (Bipartite Common Neighbors)
    paths_3 = 0
    u_neighbors = list(G.neighbors(u))
    for nj in u_neighbors:
        for oc in G.neighbors(nj):
            if G.has_edge(oc, v):
                paths_3 += 1
                
    deg_cv = G.degree(u)
    deg_job = G.degree(v)
    
    # Community match
    # Usually jobs don't have communities in the same space as CVs unless projected.
    # For now, let's just use deg/paths.
    
    return {
        "score_pa": pa,
        "score_paths_3": paths_3,
        "degree_cv": deg_cv,
        "degree_job": deg_job
    }

def build_refined_dataset(G, df_edges, cv_emb, job_emb, index):
    """
    Builds a dataset with 1:1:1 ratio (Positives : Random Negs : Hard Negs)
    """
    df_cv = pd.read_parquet("data/processed/cv.parquet")
    df_job = pd.read_parquet("data/processed/job.parquet")
    
    cv_ids = df_cv['cv_id'].tolist()
    job_ids = df_job['job_id'].tolist()
    pos_pairs = set(zip(df_edges['cv_id'], df_edges['job_id']))
    
    data = []
    
    # 1. Positives
    print(f"Processing {len(df_edges)} positive samples...")
    for _, row in df_edges.iterrows():
        c_id, j_id = row['cv_id'], row['job_id']
        idx_c = df_cv.index[df_cv['cv_id'] == c_id][0]
        idx_j = df_job.index[df_job['job_id'] == j_id][0]
        
        # Semantic Score (Cosine)
        score_sem = np.dot(cv_emb[idx_c], job_emb[idx_j])
        
        feat = extract_structural_features(c_id, j_id, G)
        data.append({**feat, "score_sem": score_sem, "label": 1})
        
    # 2. Hard Negatives (Top-K non-edges)
    print("Sampling hard negatives...")
    for i, c_id in enumerate(cv_ids):
        # Search Top 5
        try:
            D, I = index.search(cv_emb[i:i+1].astype('float32'), 5)
            scores, indices = D[0], I[0]
        except Exception as e:
            # Fallback to manual dot product
            scores_all = np.dot(cv_emb[i], job_emb.T)
            # Get top 5
            indices = np.argsort(scores_all)[::-1][:5]
            scores = scores_all[indices]
            
        for score, j_idx in zip(scores, indices):
            j_id = job_ids[j_idx]
            if (c_id, j_id) not in pos_pairs:
                feat = extract_structural_features(c_id, j_id, G)
                data.append({**feat, "score_sem": float(score), "label": 0})
                break # Just 1 hard negative per CV for balance
                
    # 3. Random Negatives
    print("Sampling random negatives...")
    n_needed = len(df_edges) # Balance 1:1 with positives
    # (Actually we have some hard negs now, let's just reach total negs = n_pos)
    current_negs = len(data) - len(df_edges)
    while current_negs < len(df_edges):
        c = np.random.choice(cv_ids)
        j = np.random.choice(job_ids)
        if (c, j) not in pos_pairs:
            idx_c = df_cv.index[df_cv['cv_id'] == c][0]
            idx_j = df_job.index[df_job['job_id'] == j][0]
            score_sem = np.dot(cv_emb[idx_c], job_emb[idx_j])
            feat = extract_structural_features(c, j, G)
            data.append({**feat, "score_sem": score_sem, "label": 0})
            current_negs += 1
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    mlflow.set_experiment("CV-Job_Link_Prediction")
    
    with mlflow.start_run():
        G = nx.read_graphml("results/G_bipartite_with_communities.graphml")
        for n, d in G.nodes(data=True): d['bipartite'] = int(d['bipartite'])
        
        cv_emb = np.load("data/embeddings/cv_embeddings.npy")
        job_emb = np.load("data/embeddings/job_embeddings.npy")
        index = faiss.read_index("data/embeddings/job_index.faiss")
        df_edges = pd.read_parquet("data/processed/edges.parquet")
        
        df = build_refined_dataset(G, df_edges, cv_emb, job_emb, index)
        
        X = df.drop(columns=['label'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training Logistic Regression fusion model...")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"ROC-AUC: {auc:.4f}")
        mlflow.log_metric("auc", auc)
        mlflow.log_params({"model_type": "LogisticRegression", "sampling": "Hard+Random"})
        
        # Save model
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fusion_link_predictor.joblib")
        mlflow.log_artifact("models/fusion_link_predictor.joblib")
        
        print("Run complete. Results logged to MLflow.")
