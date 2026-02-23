# Semantic Sentiment Classification: Neural Embedding Pipeline

This repository contains a production-grade sentiment analysis engine built during the **NIAT GenAI Masterclass**. The project demonstrates the transition from traditional keyword-based analysis to **Semantic Vector Space** classification using Transformer-based embeddings.

---

## Project Overview
The objective was to classify a dataset of **27,000 tweets** into Positive, Negative, and Neutral categories. Unlike standard NLP approaches, this pipeline utilizes high-dimensional embeddings to capture the emotional intent of a message, allowing for accurate classification even in the absence of explicit "sentiment keywords."

---

## Technical Architecture

### 1. Data Processing & EDA
The raw data underwent rigorous cleaning to remove null values and noise. Mandatory exploratory data analysis (EDA) was performed to understand class balance and text length distribution.

* **Sentiment Distribution:** Visualized the frequency of classes to ensure the model wasn't biased toward the majority "Neutral" class.
* **Text Length Analysis:** Evaluated the impact of tweet length on model precision.

![Sentiment Distribution](sentiment_distribution.png)
![Tweet Length Histogram](tweet_length_histogram.png)

### 2. The Embedding Layer (Architectural Pivot)
Initially, the project utilized the Gemini `text-embedding-004` API. However, due to external API endpoint instability (404 errors), the pipeline was refactored to use a local **Sentence Transformer** (`all-mpnet-base-v2`).

* **Benefit:** This pivot removed dependency on third-party uptime and significantly increased batch-processing speeds.
* **Vector Space:** Each tweet is represented as a 768-dimensional vector, capturing deep contextual relationships between words.

### 3. Dimensionality Reduction (UMAP)
To validate the mathematical integrity of our vectors, **UMAP** (Uniform Manifold Approximation and Projection) was used to compress the 768 dimensions into a 2D plane.

![UMAP Scatter Plot](umap_scatter_plot.png)

### 4. Classification via XGBoost
The final classification was handled by **XGBoost** (Extreme Gradient Boosting). I chose this over simpler models because it excels at finding non-linear patterns within vector data, resulting in superior F1-scores for the Positive and Negative classes.

---

## Evaluation & Results
The model achieved a balanced accuracy score, with a particularly strong performance in distinguishing clear emotional polarities.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~64% |
| **Sample Size** | 5,000 Tweets |
| **Embedding Model** | MPNet (Local) |

![Classification Report or Confusion Matrix](classification_report.png)

---

## Key Observations
* **Semantic Intent:** The model correctly identified nuanced frustration (e.g., "slow internet") even without aggressive keywords.
* **Neutral Ambiguity:** The "Neutral" class remains the most complex due to the objective nature of certain technical tweets and sarcasm.
* **Scalability:** The code is modular and can be scaled to process the full 27k dataset by adjusting the sampling logic.

---

## Future Applications: E.P.I.C.
This project serves as a foundational step toward my goal of founding **E.P.I.C. (Electric Powered Induced Cars)**. Understanding "Semantic Intent" in human language is the first step toward building AI that can predict pedestrian behavior and driver sentiment in autonomous vehicle systems.

**Author:** Puttala Jeevan Kumar (G1)  
**Affiliation:** NIAT Masterclass  
**License:** MIT
