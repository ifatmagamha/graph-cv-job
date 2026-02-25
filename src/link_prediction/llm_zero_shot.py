import os
import yaml
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def predict_link_gemini(cv_text, job_text, model):
    """
    Asks Gemini to evaluate the match between a CV and a Job.
    """
    prompt = f"""
    Evaluate the match between the following CV and Job Description.
    Provide a score between 0 and 1 (0: No match, 1: Perfect match).
    Only return the numeric score.
    
    ### CV Profile:
    {cv_text}
    
    ### Job Description:
    {job_text}
    
    Score:
    """
    try:
        response = model.generate_content(prompt)
        score_str = response.text.strip()
        # Extract float from response
        score = float(''.join(c for c in score_str if c.isdigit() or c == '.'))
        return score
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return 0.5

if __name__ == "__main__":
    config = load_config()
    api_key = config['llm']['api_key']
    
    if api_key == "YOUR_GEMINI_API_KEY" or not api_key:
        print("Please set your GEMINI_API_KEY in config.yaml")
    else:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(config['llm']['model_name'])
        
        # Select a few candidate pairs to evaluate
        processed_dir = "data/processed"
        df_cv = pd.read_parquet(os.path.join(processed_dir, "cv.parquet")).sample(5)
        df_job = pd.read_parquet(os.path.join(processed_dir, "job.parquet")).sample(5)
        
        results = []
        print("Predicting matches with Gemini...")
        for _, cv in df_cv.iterrows():
            for _, job in df_job.iterrows():
                score = predict_link_gemini(cv['text_canonical'], job['text_canonical'], model)
                results.append({
                    "cv_id": cv['cv_id'],
                    "job_id": job['job_id'],
                    "llm_score": score
                })
        
        df_llm = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df_llm.to_csv("results/gemini_predictions_sample.csv", index=False)
        print("Saved Gemini predictions to results/gemini_predictions_sample.csv")
