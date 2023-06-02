# Sentiment Analysis
* This presentation focuses on sentiment analysis, specifically building a text classification model for emotion prediction based on textual data.

* The dataset used in this project consists of text samples labeled with 9 different emotions into 3 classes: negative, neutral, and positive.


# Training machine learning model
+ The Naive Bayes model is trained on the training data using the pipeline which contian on term frequency inverse document frequency (TFIDF) as vectorizer and Naive Bayes as Classifier with the best hyperparameters using Randomized Search CV.


# Evaluation
* The performance of the model is evaluated using accuracy score is 93%.


# Run app in terminal
* Open terminal in folder app and run this command
* python -m streamlit run app.py
