import networkx as nx
from networkx.algorithms import bipartite
try:
    import community as community_louvain
except ImportError:
    import community_louvain
import os

def detect_communities(G):
    """
    Projects to CV-CV graph and detects communities using Louvain.
    """
    cv_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
    
    print("Projecting to CV-CV graph...")
    # Edge exists if CVs applied for the same job
    G_cv = bipartite.projected_graph(G, cv_nodes)
    
    print("Applying Louvain community detection...")
    partition = community_louvain.best_partition(G_cv)
    
    # Add partition back to original graph nodes
    nx.set_node_attributes(G, partition, name='community_id')
    
    # For job nodes, assign a default or find most common neighbor community?
    # For now, just set to -1 or similar
    for n in G.nodes:
        if n not in partition:
            G.nodes[n]['community_id'] = -1
            
    return G, partition

if __name__ == "__main__":
    if not os.path.exists("results/G_bipartite.graphml"):
        print("Graph file not found.")
    else:
        G = nx.read_graphml("results/G_bipartite.graphml")
        for n, d in G.nodes(data=True):
            d['bipartite'] = int(d['bipartite'])
            
        G, partition = detect_communities(G)
        n_comm = len(set(partition.values()))
        print(f"Detected {n_comm} communities among CVs.")
        
        # Save updated graph
        nx.write_graphml(G, "results/G_bipartite_with_communities.graphml")
        print("Saved updated graph to results/G_bipartite_with_communities.graphml")
