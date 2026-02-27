import subprocess
import os

def run_script(script_path):
    print(f"\n--- Running: {script_path} ---")
    result = subprocess.run(["python", script_path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing {script_path}")
        return False
    return True

def main():
    # 1. Data Preprocessing
    if not run_script("src/data/preprocess.py"): return

    # 2. Graph Construction
    if not run_script("src/graph/build_graph.py"): return

    # 3. Community Detection
    if not run_script("src/graph/community_detection.py"): return

    # 4. Multilingual Embeddings & FAISS
    if not run_script("src/embeddings/generate.py"): return

    # 5. Refined Fusion Link Prediction
    if not run_script("src/link_prediction/refine_fusion.py"): return

    # 6. Graph Enrichment
    if not run_script("src/graph/enrich_refined.py"): return

    # 7. Classification Comparison
    if not run_script("src/classification/compare_classification.py"): return

    # 8. Generation of Final Plots
    if not run_script("src/evaluation/final_plots.py"): return

    print("\n Full pipeline executed successfully!")
    print("Run 'streamlit run app.py' to explore the interactive dashboard.")

if __name__ == "__main__":
    main()
