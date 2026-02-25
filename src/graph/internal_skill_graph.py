import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
from itertools import combinations

def build_skill_graph(df_cv):
    """
    Creates a Skill-Skill co-occurrence graph from CV data.
    """
    G_skill = nx.Graph()
    
    print("Extracting skill co-occurrences...")
    for _, row in df_cv.iterrows():
        skills_str = row['skills']
        if pd.isna(skills_str):
            continue
            
        # Clean and split
        skills = [s.strip().lower() for s in skills_str.split(',') if s.strip()]
        
        # Add nodes and edges for co-occurrence
        for s1, s2 in combinations(skills, 2):
            if G_skill.has_edge(s1, s2):
                G_skill[s1][s2]['weight'] += 1
            else:
                G_skill.add_edge(s1, s2, weight=1)
                
    return G_skill

if __name__ == "__main__":
    df_cv = pd.read_parquet("data/processed/cv.parquet")
    
    G_s = build_skill_graph(df_cv)
    print(f"Skill graph built: {G_s.number_of_nodes()} skills, {G_s.number_of_edges()} relationships.")
    
    # Filter for significant relationships to avoid hairballs
    threshold = 2
    G_filtered = nx.Graph([(u, v, d) for u, v, d in G_s.edges(data=True) if d['weight'] >= threshold])
    
    os.makedirs("results", exist_ok=True)
    nx.write_graphml(G_filtered, "results/G_internal_skills.graphml")
    print("Saved skill-skill graph to results/G_internal_skills.graphml")
    
    # Simple analysis
    top_skills = sorted(G_s.degree(weight='weight'), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop Skills by Co-occurrence Importance:")
    for skill, weight in top_skills:
        print(f"- {skill}: {weight}")
