# Overview
This project focuses on building machine learning models to classify news articles as real or fake.
Using a labeled dataset of true and fake news articles, we apply basic and advanced NLP techniques to embed the text and train multiple classifiers.
The goal is to accurately predict the authenticity of a news article based solely on its textual content.

Slides: https://docs.google.com/presentation/d/1O6zvA1qc9yosN8gsIyLeK7aX5m-wGIFh/edit#slide=id.p1

# Dataset
Source: Fake News Detection Dataset on Kaggle

Contents:

Fake.csv – Articles labeled as fake.

True.csv – Articles labeled as true (sourced from Reuters).

Features:
- Title
- Full article text
- Date
- Subject

# Approach
1. Data Loading and Exploration
Loaded and combined the true and fake datasets.

Added binary labels (0 for fake, 1 for real).

Explored basic statistics: class balance, missing values, etc.

2. Preprocessing
Combined the title and text fields.

Cleaned the text: removed punctuation, stopwords, and lowercased.

Split the dataset into training and testing sets (80/20).

3. Text Embedding
TF-IDF Vectorization: Turned raw text into numeric feature vectors.

BERT Embeddings (Advanced): Used pretrained BERT to create dense semantic vectors.

4. Classification Models
Baseline Models:

Logistic Regression

Random Forest Classifier

Advanced Model:

Neural Network (MLP) trained on BERT embeddings

5. Evaluation
Used accuracy, precision, recall, and F1-score for evaluation.

Visualized performance using confusion matrices.


# Credits
Dataset provided by Emine Yetim on Kaggle.

Pretrained BERT model from HuggingFace Transformers library.



