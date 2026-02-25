import networkx as nx
from pyvis.network import Network
import os

def create_interactive_graph(graph_path, output_path="results/dynamic_graph.html"):
    """
    Generates a dynamic HTML visualization using PyVis.
    """
    G = nx.read_graphml(graph_path)
    
    # Create PyVis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    
    # Configure physics for better layout
    net.force_atlas_2based()
    
    for n, d in G.nodes(data=True):
        # Color by type (CV vs Job)
        color = "#3498db" if d.get('bipartite') == '0' else "#e74c3c"
        label = f"Node {n}" # Could use specific ID if available
        title = f"Type: {'CV' if d.get('bipartite') == '0' else 'Job'}\nComm: {d.get('community_id', 'N/A')}"
        
        net.add_node(n, label=label, title=title, color=color, size=15)
        
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, alpha=0.3)
        
    net.save_graph(output_path)
    print(f"Interactive graph saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    create_interactive_graph("results/G_bipartite_with_communities.graphml")
