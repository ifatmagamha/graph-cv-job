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
    
    # Precompute Katz Centrality or similar for global prominence
    # Note: Katz centrality is for nodes, we need it for edges
    # For now, let's use Preferential Attachment and Bipartite Common Neighbors
    
    for u, v in pairs:
        if u in G and v in G:
            # 1. Preferential Attachment: deg(u) * deg(v) 
            # (Higher degrees in both increase likelihood)
            pa = G.degree(u) * G.degree(v)
            
            # 2. Bipartite Common Neighbors (Paths of length 3)
            # Neighbors of CV u are Jobs. Neighbors of those Jobs are other CVs.
            # If Job v is a neighbor of those other CVs, it's a "friend of a friend".
            # This is hard to compute directly with nx.common_neighbors.
            # Let's use: intersection of (neighbors of neighbors of u) and {v}
            # Or simply: sum of common neighbors (CVs) between Job v and Job_k (neighbors of u)
            
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
            
            # Bipartite Common Neighbors: common Job-neighbors of CV u and some CV' that likes Job v?
            # Standard approach: look at number of paths of length 3 between u and v
            paths_len_3 = 0
            for neighbor_job in u_neighbors:
                # paths: u -> neighbor_job -> other_cv -> v
                # This doesn't make sense if v is a Job.
                # Correct: u -> neighbor_job -> other_cv -> v_is_job
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
