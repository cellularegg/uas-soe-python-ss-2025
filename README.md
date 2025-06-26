# 🎬 Movie Recommender System

This project is a full-featured movie recommendation system built with **Streamlit** using collaborative filtering and enhanced with **LLM-based insights**. It leverages the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/) and supports:

- **User-Based Collaborative Filtering** (FAISS-accelerated, dynamic)
- **Item-Based Collaborative Filtering** (LensKit, pretrained)
- **LLM Explanations** with [Ollama](https://ollama.com)

---

## 📁 Project Structure
streamlit_adjusted/
├── Home.py
├── pages/
│ └── Recommendation Models.py
├── models/
│ ├── itemBasedCollaborativeFiltering.py
│ └── userBasedCollaborativeFiltering.py
├── utils_adjusted.py
├── models/item-based-collaborative-filtering.pkl
data/
├── *.csv (MovieLens 25M files)

---

## 🚀 Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/cellularegg/uas-soe-python-ss-2025.git
cd uas-soe-python-ss-2025/streamlit_adjusted
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```
If requirements.txt doesn't work as intented:
```bash
pip install streamlit pandas numpy scikit-learn requests tqdm faiss-cpu lenskit ollama
```
### 3. Environment Setup

export TMDB_API_TOKEN=your_tmdb_token_here
export MR_RANDOM_MOVIES_COUNT=10
export MR_CACHE_POSTERS_URLS=.cache/posters.csv

TMDB_API_TOKEN is necessary for movie poster to be shown in the app.

### 4. Start Ollama (for LLM)
```bash
ollama run llama3
```
## Run the App
```bash
streamlit run Home.py
```

 Pretrained Model

Ensure you have extracted:
models/item-based-collaborative-filtering.pkl.zip
to:
models/item-based-collaborative-filtering.pkl

Also, models/model.py is required and must be present as it defines a shared interface for itemBasedCollaborativeFiltering.

## Model Overview
### User-Based Collaborative Filtering (FAISS)
Dynamic FAISS index built at runtime.

Filters neighbors by:

≥ 3 shared movie ratings (min_overlap)

≥ 15 users (min_neighbors)

Computes cosine similarity, predicts ratings using weighted aggregation.

### Item-Based Collaborative Filtering (LensKit)
Pretrained using ItemItem model from LensKit.

Recommends items similar to user-rated items.

Lightweight inference via .pkl file.

### LLM Explanations (LLaMA 3)
llama3 via Ollama

Generates human-readable summaries for each recommendation.

Requires ollama run llama3 to be active.


## Evaluation (User-Based Model)

| Evaluation | Result (1K users)   | Result (5K users)   |
|------------|---------------------|---------------------|
| RMSE       | 0.9618 ± 0.0325     | 0.9317 ± 0.0131     |

## Evaluation (Other Models)

Metrics Used:
Precision@K
Recall@K
F1@K
Hit Rate@K
NDCG@K
Coverage

## Dataset
MovieLens 25M by GroupLens
Files:
movies.csv
ratings.csv
links.csv
tags.csv
genome-scores.csv
genome-tags.csv


