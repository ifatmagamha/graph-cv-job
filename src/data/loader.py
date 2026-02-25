import pandas as pd
import os

def load_raw_data(data_config):
    """
    Loads raw CSV data from the paths specified in the config.
    """
    cv_path = data_config['raw_cv_path']
    job_path = data_config['raw_job_path']
    edges_path = data_config['raw_edges_path']
    
    print(f"Loading CVs from {cv_path}...")
    df_cv = pd.read_csv(cv_path)
    
    print(f"Loading Jobs from {job_path}...")
    df_job = pd.read_csv(job_path)
    
    print(f"Loading Edges from {edges_path}...")
    df_edges = pd.read_csv(edges_path)
    
    return df_cv, df_job, df_edges

if __name__ == "__main__":
    # Test loading with hardcoded relative paths for now
    config = {
        'raw_cv_path': "data/raw/cv_raw.csv",
        'raw_job_path': "data/raw/job_raw.csv",
        'raw_edges_path': "data/raw/matches_raw.csv"
    }
    cv, job, edges = load_raw_data(config)
    print(f"Loaded {len(cv)} CVs, {len(job)} Jobs, and {len(edges)} edges.")
