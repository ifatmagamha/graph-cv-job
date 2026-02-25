import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

def generate_embeddings(df, text_col, model_name='paraphrase-multilingual-mpnet-base-v2'):
    """
    Generates normalized embeddings for a given dataframe and text column.
    """
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    texts = df[text_col].tolist()
    print(f"Encoding {len(texts)} texts with normalization...")
    # normalize_embeddings=True ensures embeddings are on the unit hypersphere (cosine space)
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    return embeddings

if __name__ == "__main__":
    # Load processed data from Parquet
    processed_dir = "data/processed"
    df_cv = pd.read_parquet(os.path.join(processed_dir, "cv.parquet"))
    df_job = pd.read_parquet(os.path.join(processed_dir, "job.parquet"))
    
    # Generate CV embeddings
    print("Generating CV embeddings...")
    cv_embeddings = generate_embeddings(df_cv, 'text_canonical')
    
    # Generate Job embeddings
    print("Generating Job embeddings...")
    job_embeddings = generate_embeddings(df_job, 'text_canonical')
    
    # Save embeddings
    emb_dir = "data/embeddings"
    os.makedirs(emb_dir, exist_ok=True)
    np.save(os.path.join(emb_dir, "cv_embeddings.npy"), cv_embeddings)
    np.save(os.path.join(emb_dir, "job_embeddings.npy"), job_embeddings)
    
    # Build FAISS index for jobs
    print("Building FAISS index for jobs...")
    d = job_embeddings.shape[1]
    # IndexFlatIP is used for Inner Product, which is equivalent to Cosine similarity on normalized vectors.
    index = faiss.IndexFlatIP(d)
    index.add(job_embeddings.astype('float32'))
    
    faiss.write_index(index, os.path.join(emb_dir, "job_index.faiss"))
    
    print(f"Embeddings and FAISS index saved to {emb_dir}")
