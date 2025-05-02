# Overview
This project focuses on building machine learning models to classify news articles as real or fake.
Using a labeled dataset of true and fake news articles, we apply basic and advanced NLP techniques to embed the text and train multiple classifiers.
The goal is to accurately predict the authenticity of a news article based solely on its textual content.

[Slides](https://docs.google.com/presentation/d/1O6zvA1qc9yosN8gsIyLeK7aX5m-wGIFh/edit#slide=id.p1)

# Dataset
Source: [Fake News Detection Dataset on Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data)
- Fake.csv – Articles labeled as fake.
- True.csv – Articles labeled as true (sourced from Reuters).

Features:
- Title
- Full article text
- Date
- Subject

# Approach
Data Exploration
- Merged both datasets (~20,000 articles total)
- Checked for missing values and class balance (roughly 50/50 split)
- Found stylistic differences between fake and real news (e.g., article length, vocabulary complexity)

Preprocessing
- Combined title and article body into one field for richer context
- Cleaned text using NLTK: lowercasing, removing punctuation and stopwords
- Applied an 80/20 train-test split

# Text Embedding Methods
1. TF-IDF Vectorization
- Bag-of-words approach capturing word importance
- Used to train baseline models

2. BERT-Style Embeddings (MiniLM)
- Used sentence-transformers with all-MiniLM-L6-v2 to create dense, contextual embeddings of article text
- Captures semantic meaning and sentence-level context
- Trained separately on a smaller subset due to computational constraints

# Models Trained
Baseline Models (on TF-IDF)
- Logistic Regression: ~98.7% accuracy
- Random Forest: ~99.7% accuracy (potential overfitting observed)

Advanced Model (on BERT/MiniLM Embeddings)
- Multilayer Perceptron (MLP) Neural Network
- Achieved ~90% accuracy on a limited training sample (300 train, 100 test)

# Evaluation
- Used accuracy, precision, recall, and F1-score for model comparison
- Visualized predictions with confusion matrices
- Found that traditional models with TF-IDF performed surprisingly well, while MiniLM provided context-aware understanding with room for fine-tuning

# Key Takeaways
- Even simple models like Logistic Regression can be highly effective on clean, structured textual data
- BERT-style models (like MiniLM) show promise, especially when scaled and fine-tuned
- Preprocessing and smart feature engineering (like combining title + text) play a huge role in performance

# Future Work
- Fine-tune the BERT model on this specific dataset
- Scale the neural model to run on the full dataset (not just a sample)
- Explore claim-level or sentence-level fake news detection
- Deploy the system in a simple web interface



# Credits
Dataset provided by Emine Yetim on Kaggle.

Pretrained BERT model from HuggingFace Transformers library.



