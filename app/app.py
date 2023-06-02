import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Load the pre-trained model
model = joblib.load('../models/pipeline_tfidf_nb_30_may_2023.pkl')

# Function to preprocess the input text
def preprocess_text(text):
    # Load stopwords and define stemmer and lemmatizer
    stop_words = stopwords.words("english")
    excluding = ['againts', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'has', "hasn't", 'haven', "haven't", 'isn',
                "isn't", 'might', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shouldn', "shouldn't",
                'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    stop_words = [word for word in stop_words if word not in excluding]

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    cleaned_text = []
    tokens = word_tokenize(text.lower())
    
    for token in tokens:
        if (not token.isnumeric()) and (len(token) > 2) and (token not in stop_words):
            cleaned_text.append(stemmer.stem(token) and lemmatizer.lemmatize(token))
            
    cleaned_text = ' '.join(cleaned_text)
    return cleaned_text

# Function to predict the sentiment and probability
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict([preprocessed_text])
    probability = model.predict_proba([preprocessed_text])[0]
    return prediction, probability

# Define the main function for the Streamlit app
def main():
    # Set Streamlit app title and layout
    st.set_page_config(page_title="Sentiment Analysis", layout="centered")

    # Title and description
    st.title("Sentiment Analysis")
    st.write("Enter a sentence to predict its sentiment.")

    # Text input
    text_input = st.text_input("Enter a text:")
    prediction_button = st.button("Predict")

    # Perform prediction when button is clicked
    if prediction_button:
        st.subheader("Text:")
        st.write(text_input)
        prediction, probability = predict_sentiment(text_input)
        st.subheader("Prediction:")
        if prediction == 1:
            st.write("Positive: 😂")
        elif prediction == 0:
            st.write("Neutral: 😐")
        else:
            st.write("Negative: 😔")
        st.subheader("Probability:")
        st.write(f"Probability of Sad: {probability[0]:.2f}")
        st.write(f"Probability of Neutral: {probability[1]:.2f}")
        st.write(f"Probability of Happy: {probability[2]:.2f}")

        # Plot countplot for probability
        data = {"Probability": ['Sad' , 'Neutral', 'Happy'], "Value": probability}
        df = pd.DataFrame(data)
        fig, ax = plt.subplots()
        sns.barplot(x="Probability", y="Value", data=df)
        ax.set_title("Probability")
        
        # Add annotations to the bars
        for p in ax.patches:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(f'{y:.2f}', (x, y), ha='center', va='bottom')
        
        st.pyplot(fig)

if __name__ == '__main__':
    main()
