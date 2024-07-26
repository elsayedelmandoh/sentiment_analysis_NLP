# Sentiment Analysis

Developed a sentiment analysis to classify text as positive, negative, or neutral. Utilized NLTK for preprocessing and fine-tuned the TF-IDF vectorizer and Naive Bayes classifier with RandomizedSearchCV, achieving 93% validation accuracy and 91% testing accuracy. Additionally, implemented an LSTM model achieving 92% validation and testing.

## Table of Contents

- [Overview](#overview)
- [Run app in terminal](#run-app-in-terminal)
- [Contributing](#contributing)
- [Author](#author)

## Overview

* This presentation focuses on sentiment analysis, specifically building a text classification model for emotion prediction based on textual data.

* The dataset used in this project consists of text samples labeled with 9 different emotions into 3 classes: negative, neutral, and positive.

* The Naive Bayes model is trained on the training data using the pipeline which contian on term frequency inverse document frequency (TFIDF) as vectorizer and Naive Bayes as Classifier with the best hyperparameters using Randomized Search CV.


## Run app in terminal
* Open terminal in folder app and run this command
* python -m streamlit run app.py

## Contributing

Contributions are welcome! If you have suggestions, improvements, or additional content to contribute, feel free to open issues, submit pull requests, or provide feedback. 

[![GitHub watchers](https://img.shields.io/github/watchers/elsayedelmandoh/sentiment_analysis_NLP.svg?style=social&label=Watch)](https://GitHub.com/elsayedelmandoh/sentiment_analysis_NLP/watchers/?WT.mc_id=academic-105485-koreyst)
[![GitHub forks](https://img.shields.io/github/forks/elsayedelmandoh/sentiment_analysis_NLP.svg?style=social&label=Fork)](https://GitHub.com/elsayedelmandoh/sentiment_analysis_NLP/network/?WT.mc_id=academic-105485-koreyst)
[![GitHub stars](https://img.shields.io/github/stars/elsayedelmandoh/sentiment_analysis_NLP.svg?style=social&label=Star)](https://GitHub.com/elsayedelmandoh/sentiment_analysis_NLP/stargazers/?WT.mc_id=academic-105485-koreyst)

## Author

This repository is maintained by Elsayed Elmandoh, an AI Engineer. You can connect with Elsayed on [LinkedIn and Twitter/X](https://linktr.ee/elsayedelmandoh) for updates and discussions related to Machine learning, deep learning and NLP.

Happy coding!
