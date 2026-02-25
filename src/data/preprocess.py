import pandas as pd
import re
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters but keep some structure
    text = re.sub(r'[^a-zA-Z0-9\s,.-]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_cv(df_cv):
    df_cv = df_cv.copy()
    # Concat fields for canonical text
    # cv_id,text,skills,level,profile_type,domain
    df_cv['text_canonical'] = df_cv['text'].fillna('') + " " + df_cv['skills'].fillna('')
    df_cv['text_canonical'] = df_cv['text_canonical'].apply(clean_text)
    return df_cv

def preprocess_job(df_job):
    df_job = df_job.copy()
    # job_id,text,skills,level_required,domain
    df_job['text_canonical'] = df_job['text'].fillna('') + " " + df_job['skills'].fillna('')
    df_job['text_canonical'] = df_job['text_canonical'].apply(clean_text)
    return df_job

if __name__ == "__main__":
    from loader import load_raw_data
    
    config = {
        'raw_cv_path': "data/raw/cv_raw.csv",
        'raw_job_path': "data/raw/job_raw.csv",
        'raw_edges_path': "data/raw/matches_raw.csv"
    }
    df_cv, df_job, df_edges = load_raw_data(config)
    
    print("Preprocessing CVs...")
    df_cv_clean = preprocess_cv(df_cv)
    print("Preprocessing Jobs...")
    df_job_clean = preprocess_job(df_job)
    
    # Save processed data
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    df_cv_clean.to_parquet(os.path.join(processed_dir, "cv.parquet"), index=False)
    df_job_clean.to_parquet(os.path.join(processed_dir, "job.parquet"), index=False)
    df_edges.to_parquet(os.path.join(processed_dir, "edges.parquet"), index=False)
    
    print(f"Saved cleaned data in Parquet format to {processed_dir}/")
