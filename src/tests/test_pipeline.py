import os
import pandas as pd
import numpy as np
import networkx as nx
import pytest

def test_processed_data_exists():
    assert os.path.exists("data/processed/cv.parquet")
    assert os.path.exists("data/processed/job.parquet")
    assert os.path.exists("data/processed/edges.parquet")

def test_embeddings_integrity():
    cv_emb = np.load("data/embeddings/cv_embeddings.npy")
    job_emb = np.load("data/embeddings/job_embeddings.npy")
    assert cv_emb.shape[1] == 768 # MPNet dimension
    assert job_emb.shape[1] == 768
    # Check normalization
    assert np.allclose(np.linalg.norm(cv_emb[0]), 1.0, atol=1e-5)

def test_graph_structure():
    assert os.path.exists("results/G_bipartite_with_communities.graphml")
    G = nx.read_graphml("results/G_bipartite_with_communities.graphml")
    # Must have bipartite attribute as string '0' or '1' or int 0 or 1
    for n, d in G.nodes(data=True):
        assert 'bipartite' in d
        assert str(d['bipartite']) in ['0', '1']

def test_model_loading():
    import joblib
    assert os.path.exists("models/fusion_link_predictor.joblib")
    model = joblib.load("models/fusion_link_predictor.joblib")
    assert hasattr(model, "predict_proba")
