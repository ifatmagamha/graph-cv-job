import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def evaluate_semantic_matching(cv_embeddings, job_embeddings, df_edges, top_k=10):
    """
    Evaluates semantic matching by checking if the ground truth job is in the top-K.
    """
    print(f"Computing similarity matrix of shape ({len(cv_embeddings)}, {len(job_embeddings)})...")
    sim_matrix = cosine_similarity(cv_embeddings, job_embeddings)
    
    # Load metadata to map indices
    df_cv = pd.read_csv("data/processed/cv_clean.csv")
    df_job = pd.read_csv("data/processed/job_clean.csv")
    
    cv_id_to_idx = {id: i for i, id in enumerate(df_cv['cv_id'])}
    job_id_to_idx = {id: i for i, id in enumerate(df_job['job_id'])}
    
    hits = 0
    total = 0
    
    results = []
    
    print("Evaluating matches...")
    for _, row in df_edges.iterrows():
        cv_id = row['cv_id']
        job_id = row['job_id']
        
        if cv_id in cv_id_to_idx and job_id in job_id_to_idx:
            cv_idx = cv_id_to_idx[cv_id]
            gt_job_idx = job_id_to_idx[job_id]
            
            # Get similarities for this CV across all jobs
            cv_sims = sim_matrix[cv_idx]
            
            # Find top-K job indices
            top_k_indices = np.argsort(cv_sims)[-top_k:][::-1]
            
            if gt_job_idx in top_k_indices:
                hits += 1
            total += 1
            
            results.append({
                "cv_id": cv_id,
                "job_id": job_id,
                "top_k_indices": top_k_indices.tolist(),
                "gt_rank": list(np.argsort(cv_sims)[::-1]).index(gt_job_idx) + 1
            })
            
    hit_rate = hits / total if total > 0 else 0
    return hit_rate, results

if __name__ == "__main__":
    # Load embeddings and edges
    cv_emb = np.load("data/embeddings/cv_embeddings.npy")
    job_emb = np.load("data/embeddings/job_embeddings.npy")
    df_edges = pd.read_csv("data/processed/edges.csv")
    
    hit_rate, results = evaluate_semantic_matching(cv_emb, job_emb, df_edges)
    
    print(f"Semantic Link Prediction Hit@{10}: {hit_rate:.4f}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/semantic_eval.csv", index=False)
    print("Saved evaluation to results/semantic_eval.csv")
