import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
import numpy as np
import os

def compute_refined_structural_metrics(G, pairs):
    """
    Computes bipartite-compatible structural scores for CV-Job pairs.
    pairs: list of (cv_id, job_id)
    """
    results = []
    print(f"Computing structural scores for {len(pairs)} pairs...")
    
    for u, v in pairs:
        if u in G and v in G:
            # 1. Preferential Attachment: deg(u) * deg(v) 
            # (Higher degrees in both increase likelihood)
            pa = G.degree(u) * G.degree(v)
            
            # 2. Bipartite Common Neighbors (Paths of length 3)
            # simply: sum of common neighbors (CVs) between Job v and Job_k (neighbors of u)
            
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
            
            # Bipartite Common Neighbors: common Job-neighbors of CV u and some CV' that likes Job v?
            # Standard approach: look at number of paths of length 3 between u and v
            paths_len_3 = 0
            for neighbor_job in u_neighbors:
                for other_cv in G.neighbors(neighbor_job):
                    if G.has_edge(other_cv, v):
                        paths_len_3 += 1
            
            results.append({
                "cv_id": u,
                "job_id": v,
                "score_pa": pa,
                "score_paths_3": paths_len_3,
                "degree_cv": G.degree(u),
                "degree_job": G.degree(v),
                "comm_cv": G.nodes[u].get('community_id', -1)
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    if not os.path.exists("results/G_bipartite_with_communities.graphml"):
        print("Graph not found.")
    else:
        G = nx.read_graphml("results/G_bipartite_with_communities.graphml")
        for n, d in G.nodes(data=True):
            d['bipartite'] = int(d['bipartite'])
            
        df_edges = pd.read_parquet("data/processed/edges.parquet")
        pairs = list(zip(df_edges['cv_id'], df_edges['job_id']))
        
        df_struct = compute_refined_structural_metrics(G, pairs)
        
        os.makedirs("results", exist_ok=True)
        df_struct.to_csv("results/structural_features_matches.csv", index=False)
        print("Structural features saved to results/structural_features_matches.csv")
