import networkx as nx
import pandas as pd
import os

def build_bipartite_graph(df_cv, df_job, df_edges):
    """
    Constructs a bipartite graph using NetworkX.
    Nodes are assigned a 'bipartite' attribute (0 for CVs, 1 for Jobs).
    """
    G = nx.Graph()
    
    # Add CV nodes
    for _, row in df_cv.iterrows():
        G.add_node(row['cv_id'], bipartite=0, **row.drop('cv_id').to_dict())
        
    # Add Job nodes
    for _, row in df_job.iterrows():
        G.add_node(row['job_id'], bipartite=1, **row.drop('job_id').to_dict())
        
    # Add Edges
    for _, row in df_edges.iterrows():
        if row['cv_id'] in G and row['job_id'] in G:
            G.add_edge(row['cv_id'], row['job_id'], label=row['label'])
            
    return G

if __name__ == "__main__":
    # Load cleaned data
    df_cv = pd.read_csv("data/processed/cv_clean.csv")
    df_job = pd.read_csv("data/processed/job_clean.csv")
    df_edges = pd.read_csv("data/processed/edges.csv")
    
    print("Building bipartite graph...")
    G = build_bipartite_graph(df_cv, df_job, df_edges)
    
    print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print(f"Is bipartite: {nx.is_bipartite(G)}")
    
    # Save graph
    os.makedirs("results", exist_ok=True)
    nx.write_graphml(G, "results/G_bipartite.graphml")
    print("Saved graph to results/G_bipartite.graphml")
