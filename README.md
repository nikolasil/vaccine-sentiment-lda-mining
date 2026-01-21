# Vaccine Sentiment Analysis & Topic Modeling 🧬📊

This project provides a comprehensive computational analysis of COVID-19 vaccination discourse on Twitter. It features a full pipeline from raw data preprocessing and exploratory data analysis (EDA) to supervised classification and unsupervised topic modeling.

## 🚀 Project Overview
The goal is to identify and categorize public opinion regarding COVID-19 vaccines into Positive, Negative, or Neutral sentiments, while simultaneously discovering latent discussion topics using advanced clustering techniques.

## 🛠️ Technical Pipeline
### Robust Data Preprocessing
- Social media text is noisy. I implemented a modular cleaning pipeline to ensure high-quality input for the models:
- Noise Reduction: Removal of URLs, hashtags, punctuation, and extra whitespace.
- Linguistic Normalization: Lowercasing and English stopword removal using NLTK.
- Tokenization & Cleaning: Handling emojis and pictographs via Regex patterns to focus on semantic text.
### Exploratory Data Analysis (EDA)
- Analysis of the vaccination dataset revealed a strong lean toward neutral reporting and specific brand mentions:
- Class Distribution: Neutral: ~75.0%, Negative: ~13.5%, Positive: ~11.5%
- Keyword Insights: The most frequent terms identified across the dataset include covaxin, moderna, pfizer, and slots (referring to appointment availability).

### Feature Engineering
I compared three distinct vectorization strategies to represent text numerically:

- Bag-of-Words (CountVectorizer): Simple frequency counts.
- TF-IDF (TfidfVectorizer): Weighted importance to penalize common words.
- Word Embeddings (Gensim): Dense vector representations to capture semantic relationships.
- 
### Machine Learning & Topic Modeling
Supervised Learning: Implementation of SVM, Random Forest, and KNN classifiers.

Topic Modeling (LDA): Used Latent Dirichlet Allocation to uncover hidden themes. I utilized Topic Coherence scores to mathematically determine the optimal number of clusters and pyLDAvis for interactive visualization of the topic space.

## 📈 Key Insights
Brand Sentiment: Analysis showed that Astrazeneca discussions had a higher negative sentiment ratio (~20.0%) compared to the Moderna/Pfizer/Biontech group (~18.3%).

Temporal Trends: The data suggests public discourse is heavily driven by vaccine "slots" and "availability" rather than just clinical opinion.

## 🗂️ Project Structure
analysis.ipynb: The primary Jupyter Notebook containing the full implementation.

data/: Directory for raw and processed .tsv or .pkl files.

models/: Serialized models and vectorizers using pickle.

## 🎓 Academic Context
Developed as part of the Data Mining course at the National and Kapodistrian University of Athens (UoA). Based on the UC Berkeley CS188 framework.

## ⚙️ Requirements
Bash

pip install pandas numpy scikit-learn gensim nltk matplotlib torch pyLDAvis
