# Graph-based CV-Job Mining Project

Ce projet vise à modéliser et analyser les interactions entre Curriculum Vitae (CV) et Offres d'Emploi (Jobs) sous forme de graphe biparti. L'objectif final est d'améliorer le matching via des techniques de link prediction et de node classification enrichies par des LLM.

## Architecture Globale

Le pipeline suit une approche hybride combinant :
1. **Sémantique** : Embeddings LLM (sentence-transformers) pour capturer le sens profond des textes.
2. **Structurel** : Métriques de graphes (Centralité, Communautés) pour capturer les relations topologiques.

## Résultats Finaux

### 1. Analyse de Graphe
- **Nodes** : 390 (CVs + Jobs)
- **Edges** : 1200 (Matches originaux) + 500 (Prédictions à haute confiance)
- **Communautés** : 6 clusters détectés (représentant les pôles de compétences).

### 2. Link Prediction (Matching)
| Modèle | Performance |
| :--- | :--- |
| **Sémantique (Hit@10)** | 54.17% |
| **Fusion Hybride (ROC-AUC)** | 86.45% |

### 3. Node Classification
- **Cible** : Domaine d'activité du candidat.
- **Accuracy** : **95.00%**.

## Structure du Projet
- `src/data` : Préparation, nettoyage et chargement.
- `src/graph` : Construction, métriques, détection de communautés et enrichissement.
- `src/embeddings` : Génération de vecteurs via `all-MiniLM-L6-v2`.
- `src/link_prediction` : Algorithmes classiques, sémantiques et modèle de fusion (Random Forest).
- `src/classification` : Classification de nœuds basée sur les features structurelles et sémantiques.
- `results/` : Dossier contenant les graphes exportés (GraphML) et les rapports de performance.

## Comment reproduire
1. Installer les dépendances : `pip install -r requirements.txt`
2. Lancer le pipeline complet : `python main.py`
3. Lancer le dashboard interactif : `streamlit run app.py`

## Robustesse & Tests
Pour vérifier l'intégrité du système :
`python -m pytest src/tests/test_pipeline.py`

## Structure du Projet (Clean)
- `src/data` : Préprocessing Parquet.
- `src/graph` : Construction, Communautés et Enrichissement.
- `src/embeddings` : `paraphrase-multilingual-mpnet-base-v2` + FAISS.
- `src/link_prediction` : Fusion supervisée (Logistic Regression + MLflow).
- `src/classification` : Études comparatives avant/après enrichissement.
- `app.py` : Dashboard Streamlit.
- `main.py` : Orchestrateur.
