import networkx as nx
from networkx.algorithms import bipartite
import json
import os

def compute_metrics(G):
    """
    Computes key structural metrics for the bipartite graph.
    """
    metrics = {}
    
    # 1. Degree Stats
    cv_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    job_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 1]
    
    metrics['n_cv'] = len(cv_nodes)
    metrics['n_job'] = len(job_nodes)
    metrics['n_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Degree centrality (normalized)
    deg_centrality = nx.degree_centrality(G)
    metrics['avg_degree_cv'] = sum(G.degree(n) for n in cv_nodes) / len(cv_nodes)
    metrics['avg_degree_job'] = sum(G.degree(n) for n in job_nodes) / len(job_nodes)
    
    # Closeness Centrality (Bipartite optimized)
    # Using generic for now
    closeness = nx.closeness_centrality(G)
    metrics['top_5_closeness'] = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Connectivity
    metrics['is_connected'] = nx.is_connected(G)
    if not metrics['is_connected']:
        metrics['n_components'] = nx.number_connected_components(G)
        
    return metrics

if __name__ == "__main__":
    if not os.path.exists("results/G_bipartite.graphml"):
        print("Graph file not found. Run build_graph.py first.")
    else:
        G = nx.read_graphml("results/G_bipartite.graphml")
        # Restore bipartite attribute (graphml saves it as int string or bool)
        for n, d in G.nodes(data=True):
            d['bipartite'] = int(d['bipartite'])
            
        print("Computing structural metrics...")
        metrics = compute_metrics(G)
        
        os.makedirs("results", exist_ok=True)
        with open("results/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        print("Results saved to results/metrics.json")
        print(f"Average Degree (CV): {metrics['avg_degree_cv']:.2f}")
        print(f"Average Degree (Job): {metrics['avg_degree_job']:.2f}")
