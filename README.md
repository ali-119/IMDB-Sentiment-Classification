# IMDB Sentiment Classification Project
Determine the IMDB comment class type

# Overview
This project focuses on predicting **movie review sentiments** — determining whether a review is *positive* or *negative*.
It demonstrates how text data can be transformed into numerical vectors and classified using machine learning models for **Natural Language Processing (NLP)** tasks.

------

# Dataset
- **Name:** IMDB Movie Reviews Dataset
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **Size:** 50,000 records (balanced: 25,000 positive and 25,000 negative)
- **Content:** Each record includes a movie review (review) and its sentiment label (*positive* or *negative*).

-----

# Project Workflow
## Data Preparation
- Loaded the dataset using pandas
- Checked for missing or empty reviews
- Verified class balance between positive and negative samples
- Sampled 4% (≈1,000 reviews) of the dataset for faster model comparison

## Exploratory Text Analysis
- Used CountVectorizer to extract the top 20 most frequent non-stop words for each sentiment
- Compared vocabulary patterns between positive and negative reviews

## Model Training
- Built and evaluated two text classification pipelines:
- TF-IDF + LinearSVC
- Converts raw text to TF-IDF representation
- Trains a Support Vector Classifier for robust classification
- TF-IDF + MultinomialNB
- Uses a Naive Bayes classifier with TF-IDF features
- Lightweight and effective for text-based tasks
- Both pipelines were implemented using scikit-learn’s Pipeline API.

## Model Evaluation
### Models were compared using:

- classification_report → Accuracy, Precision, Recall, F1-score
- ConfusionMatrixDisplay → Visual evaluation of prediction accuracy Results
- TF-IDF + LinearSVC: Achieved higher precision and accuracy (~86%)
- TF-IDF + MultinomialNB: Slightly lower accuracy (~74%)

### Key Insights:
- Positive reviews commonly used words like great, love, best, story
- Negative reviews frequently contained bad, worst, boring, waste
- TF-IDF representation improved generalization by minimizing bias
from frequent but uninformative words.

-----

# Libraries Used
- pandas
- matplotlib
- scikit-learn

-----

# Future Improvements
- Add GridSearchCV for hyperparameter tuning
- Enhance text preprocessing (lemmatization, punctuation removal)
- Experiment with Logistic Regression and Ensemble Models
- Extend with Word2Vec, GloVe, or BERT embeddings

-----

# How to Run
## Clone the repository:
git clone [github](https://github.com/ali-119/IMDB-Sentiment-Classification)
cd IMDB-Sentiment-Classification

## Install dependencies:
<pre> pip install -r requirements.txt </pre>

## open the Jupyter notebook:
jupyter notebook notebooks/IMDB_Sentiment_Classification.ipynb

Run all cells to train and evaluate the model.

-----
# Final Conclusion:
After training and evaluating both models — *LinearSVC and MultinomialNB* — 
we observed that both performed well on the balanced IMDB dataset (*positive* vs *negative* reviews).

However, LinearSVC achieved slightly higher overall accuracy and better generalization on unseen data, 
while MultinomialNB was faster but more sensitive to noise and vocabulary variations.

The TF-IDF representation effectively reduced the influence of common words and improved classification quality.
Overall, the combination of *TF-IDF + LinearSVC* can be considered the best-performing pipeline for this dataset.

-----

# Author ✍️
**Author:** Ali  
**Field:** Data Science & Machine Learning Student  
**Email:** ali.hz87980@gmail.com  
**GitHub:** [ali-119](https://github.com/ali-119)
