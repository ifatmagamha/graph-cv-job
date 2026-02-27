import pandas as pd
import numpy as np
import networkx as nx
import joblib
import os

def enrich_graph(G, model, cv_emb, job_emb, threshold=0.9):
    """
    Predicts new edges for all non-linked pairs and adds those above threshold to G.
    """
    df_cv = pd.read_parquet("data/processed/cv.parquet")
    df_job = pd.read_parquet("data/processed/job.parquet")
    
    cv_ids = df_cv['cv_id'].tolist()
    job_ids = df_job['job_id'].tolist()
    pos_pairs = set(zip(pd.read_parquet("data/processed/edges.parquet")['cv_id'], 
                        pd.read_parquet("data/processed/edges.parquet")['job_id']))
    
    new_edges = []
    print("Finding candidate pairs for enrichment...")
    
    for i, c_id in enumerate(cv_ids):
        for j, j_id in enumerate(job_ids):
            if (c_id, j_id) not in pos_pairs:
                # Extract features (same as in fusion.py)
                sem = np.dot(cv_emb[i], job_emb[j])
                
                # Structural (simplified for batch)
                pa = G.degree(c_id) * G.degree(j_id) if (c_id in G and j_id in G) else 0
                
                # We use a simple feature vector for the prediction
                # [score_pa, score_paths_3, degree_cv, degree_job, score_sem]
                if sem > 0.8: # Pre-filter for efficiency
                    # Precise features
                    paths_3 = 0
                    if c_id in G and j_id in G:
                        for nj in G.neighbors(c_id):
                            for oc in G.neighbors(nj):
                                if G.has_edge(oc, j_id):
                                    paths_3 += 1
                                    
                    feat = np.array([[pa, paths_3, G.degree(c_id), G.degree(j_id), sem]])
                    prob = model.predict_proba(feat)[:, 1][0]
                    
                    if prob >= threshold:
                        new_edges.append((c_id, j_id, prob))
                        
    # Add to G
    for u, v, p in new_edges:
        G.add_edge(u, v, weight=p, edge_type="predicted")
        
    print(f"Added {len(new_edges)} predicted edges to the graph.")
    return G, new_edges

if __name__ == "__main__":
    G = nx.read_graphml("results/G_bipartite_with_communities.graphml")
    for n, d in G.nodes(data=True): d['bipartite'] = int(d['bipartite'])
    
    model = joblib.load("models/fusion_link_predictor.joblib")
    cv_emb = np.load("data/embeddings/cv_embeddings.npy")
    job_emb = np.load("data/embeddings/job_embeddings.npy")
    
    G_enriched, enrichment_list = enrich_graph(G, model, cv_emb, job_emb)
    
    os.makedirs("results", exist_ok=True)
    nx.write_graphml(G_enriched, "results/G_enriched_refined.graphml")
    
    df_new = pd.DataFrame(enrichment_list, columns=['cv_id', 'job_id', 'prob'])
    df_new.to_csv("results/new_predicted_edges_refined.csv", index=False)
    print("Enriched graph saved to results/G_enriched_refined.graphml")
