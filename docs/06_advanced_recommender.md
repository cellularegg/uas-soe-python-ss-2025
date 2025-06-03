# Advanced Movie Recommender System

A production-ready movie recommendation system built with Python, featuring multiple recommendation approaches and hybrid model architecture for optimal performance.

## 🎯 Notebook Overview

This project implements a comprehensive movie recommender system using the MovieLens dataset, combining collaborative filtering, content-based filtering, and knowledge-based approaches through a hybrid architecture. The system achieves high accuracy while handling cold start problems and providing diverse recommendations.

## ✨ Key Features

- **Multi-Modal Hybrid Architecture**: Combines 3 distinct recommendation approaches
- **Production-Ready Performance**: Optimized for real-world deployment
- **Cold Start Handling**: Effective recommendations for new users and items
- **Comprehensive Evaluation**: Industry-standard ranking metrics (Precision@K, Recall@K, nDCG@K)
- **Scalable Design**: Efficient model persistence and fast inference
- **Interactive Web Interface**: Streamlit-based recommendation demo

## 🏗️ System Architecture

### Recommendation Models

1. **Collaborative Filtering Models**
   - **SVD (Basic)**: RMSE ~0.777 - Primary production model
   - **SVD++**: RMSE ~0.826 - Advanced implicit feedback model
   - **NMF**: RMSE ~0.886 - Non-negative matrix factorization

2. **Content-Based Filtering**
   - TF-IDF vectorization of movie genres and genome tags
   - Cosine similarity for content-based recommendations
   - Relevance threshold filtering (>0.8)

3. **Knowledge-Based Filtering**
   - Genome tag-based semantic similarity
   - Movie-tag relevance scoring
   - Semantic relationship recommendations

### Hybrid Model Configuration
```python
# Optimized hybrid weights
weights = {
    'collaborative': 0.4,  # SVD model
    'content': 0.3,        # Genre + genome tags
    'knowledge': 0.3       # Semantic similarity
}
```

## 📊 Performance Results

| Model | RMSE | Use Case | Status |
|-------|------|----------|--------|
| **SVD (Recommended)** | 0.777 | Production deployment | ✅ Ready |
| **SVD++** | 0.826 | Accuracy-critical scenarios | ✅ Available |
| **NMF** | 0.886 | Alternative approach | ✅ Available |

**Expected Ranking Performance:**
- Precision@10: 0.20 - 0.40
- Recall@10: 0.08 - 0.18  
- nDCG@10: 0.25 - 0.50

## 🔧 Technical Implementation

### Feature Engineering
- **Movie Features**: Average ratings, rating counts, genome tag profiles
- **User Features**: Rating patterns, preference modeling
- **Content Features**: TF-IDF vectors from genres and semantic tags
- **Relevance Filtering**: Minimum relevance threshold of 0.8 for quality

### Model Persistence
All models are serialized and version-controlled:
- Pickle files for scikit-learn models
- NumPy arrays for similarity matrices
- Parquet files for structured data

### Research Insights
- **Model Complexity ≠ Better Performance**: Basic SVD beat advanced models
- **Feature Engineering is Key**: Quality data preparation drives success
- **Hybrid Approach Works**: Multi-modal strategy provides robustness
- **Evaluation Framework Critical**: Proper metrics guide optimal decisions

## 🚀 Quick Start

### Development Environment
This project requires significant computational resources for optimal performance:
- **CPU**: 16 cores recommended
- **RAM**: 64 GB required for full model training
- **Codespace**: Hosted on `fhtw-dsc-inf` organization for cost management

> **Note**: Codespace usage for this notebook is paid for by `fhtw-dsc-inf` due to the high computational requirements of training multiple collaborative filtering models simultaneously.

> **Storage Limitation**: Git LFS (Large File Storage) is unfortunately disabled on FHTW Git repositories, preventing efficient version control of large model files (.pkl, .npy files >100MB) that could have provided the best of both worlds - local institutional hosting with proper large file management.

