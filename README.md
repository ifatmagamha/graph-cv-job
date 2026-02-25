# Graph-based CV-Job Mining Project

Ce projet vise à modéliser et analyser les interactions entre CV et Offres d'Emploi (Jobs) sous forme de graphe biparti. L'objectif final est d'améliorer le matching via des techniques de link prediction et de node classification enrichies par des LLM.

## Architecture Globale

Le pipeline suit une approche hybride combinant :
1. **Sémantique** : Embeddings LLM (sentence-transformers) pour capturer le sens profond des textes.
2. **Structurel** : Métriques de graphes (Centralité, Communautés) pour capturer les relations topologiques.

## Objectifs Business
Ce projet vise à révolutionner le recrutement grâce à la **Théorie des Graphes** et aux **LLM**. Les objectifs principaux sont :
1.  **Automatisation du Matching** : Prédire la pertinence d'un candidat pour un poste avec une précision supérieure aux mots-clés classiques.
2.  **Profiling Avancé** : Classifier les profils (Junior/Senior, Spécialiste/Polyvalent) via leur positionnement dans l'écosystème de compétences.
3.  **Identification des Lacunes** : Détecter les liens manquants (opportunités de carrière) qui ne sont pas visibles via une simple recherche sémantique.

## Modélisation
-   **Graphe Biparti CV-Job** : Implémenté via NetworkX avec gestion des attributs spécifiques à chaque type de nœud.
-   **Prédiction de Liens Hybride** : Fusion sémantique (Embeddings Gemini/SBERT) et structurelle (Common Neighbors, Preferential Attachment).
-   **Analyse de Communautés** :
    -   *Global* : Identification de clusters de métiers via la projection CV-CV.
    -   *Interne* : Modélisation des graphes de compétences (`src/graph/internal_skill_graph.py`) pour capturer la cohérence des profils.
-   **Classification** : Utilisation des features combinées pour catégoriser les nœuds.

## Stack Technique
-   **Graph Engine** : NetworkX, PyVis (Visualisation).
-   **Intelligence Artificielle** : Google Gemini API (Link Prediction Zero-Shot), Sentence-Transformers (Embeddings).
-   **Machine Learning** : Scikit-learn (Fusion de features, Classification), FAISS (Recherche vectorielle rapide).
-   **Tracking** : MLflow pour le suivi des expériences de Link Prediction.

## Choix des Métriques et Résultats
Nous avons sélectionné des métriques spécifiques à deux problématiques différentes :

### 1. Prédiction de Liens (Recommandation)
-   **Hits@10** (54.17%) : Capacité du système à placer le bon job dans le top 10 des recommandations.
-   **MRR (Mean Reciprocal Rank)** : Important pour favoriser les systèmes qui placent les meilleures correspondances en haut de liste.
-   **ROC-AUC** (86.45%) : Mesure la capacité du modèle de fusion à distinguer un vrai match d'un mauvais match.

### 2. Classification de Nœuds (Profiling)
-   **Accuracy** (95.00%) : Performance globale de la classification des domaines d'activité.
-   **Macro-F1 Score** : Utilisé pour assurer une performance robuste même sur les domaines moins représentés (équilibrage des classes).

## Analyse des Coûts
L'utilisation de LLM (Gemini) apporte une précision fine mais implique des coûts :
-   **Traitement Batch** : Préférable d'utiliser des embeddings locaux (SBERT) pour la recherche à large échelle.
-   **Raffinement** : Le LLM est utilisé uniquement sur les paires candidates à haut score sémantique pour minimiser les appels API et la latence.

## Défis & Perspectives
### Défis rencontrés
-   **Sparsité des données** : Le graphe initial contient peu d'arêtes par rapport au nombre total de paires possibles.
-   **Complexité Bipartite** : Les algorithmes classiques de link prediction (ex: Adamic-Adar) ont dû être adaptés pour la structure bipartite (chemins de longueur 3).

### Perspectives
-   **Graph Neural Networks (GNN)** : Passer de la régression logistique à un modèle type GraphSAGE pour un apprentissage de représentations plus profond.
-   **Enrichissement Temporel** : Intégrer l'évolution des carrières (temps passé sur un poste) comme poids dans le graphe.

## 📂 Structure du Projet
-   `src/data` : Préprocessing Parquet et extraction de skills.
-   `src/graph` : Construction, Communautés, Graphes internes et Enrichissement.
-   `src/embeddings` : Embeddings multilingues et indexation FAISS.
-   `src/link_prediction` : Fusion supervisée et scoring LLM.
-   `src/classification` : Études comparatives et profiling.
-   `main.py` : Orchestrateur du pipeline complet.

## Comment reproduire
1. Installer les dépendances : `pip install -r requirements.txt`
2. Lancer le pipeline complet : `python main.py`
3. Lancer le dashboard interactif : `streamlit run app.py`

## Robustesse & Tests
Pour vérifier l'intégrité du système :
`python -m pytest src/tests/test_pipeline.py`