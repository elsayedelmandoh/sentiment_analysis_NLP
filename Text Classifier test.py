import pandas as pd
# visualize dataset
import matplotlib.pyplot as plt
import seaborn as sns
# read training dataset
dataset_train = pd.read_csv('EA-train.txt', header=None, names= ['text', 'emotion'] ,delimiter=';')
# quoting: ???
# first five rows
dataset_train.head()
# last five rows
dataset_train.tail()
# last five rows
dataset_train.tail()
dataset_train.info()
# (rows, cols)
dataset_train.shape
# describe data
dataset_train.describe()
dataset_train.dropna(inplace=True)
dataset_train.isnull().sum()
# unique values
dataset_train['emotion'].unique()
dataset_train['emotion'].value_counts()
# plot
ax= sns.countplot(x=dataset_train['emotion'],
                  data=dataset_train)

for p in ax.patches: # bars
    '''
    get_bbox(): return bounding box of the bar, 
    get_points(): returns the coordinates of the four corners of the bounding box.
    '''
    x= p.get_bbox().get_points()[:,0] # extract the x-coordinates of the four corners of the bar rectangle
    y= p.get_bbox().get_points()[1,1] # extract the y-coordinate of the top-right corner
    ax.annotate(f'{y:.0f}', (x.mean(), y), ha='center',va='bottom') # text on top bar
    
plt.title("Emotion rating is imbalance")
plt.show()
dataset_train['label'] = dataset_train['emotion']
dataset_train['label'].replace(['joy', 'love', 'surprise'], 'happy', inplace=True)
dataset_train['label'].replace(['sadness', 'anger', 'fear'], 'sad', inplace=True)
dataset_train['label'].unique()
dataset_train['label'].value_counts()
# plot
ax= sns.countplot(x=dataset_train['label'],
                  data=dataset_train)

for p in ax.patches: # bars
    '''
    get_bbox(): return bounding box of the bar, 
    get_points(): returns the coordinates of the four corners of the bounding box.
    '''
    x= p.get_bbox().get_points()[:,0] # extract the x-coordinates of the four corners of the bar rectangle
    y= p.get_bbox().get_points()[1,1] # extract the y-coordinate of the top-right corner
    ax.annotate(f'{y:.0f}', (x.mean(), y), ha='center',va='bottom') # text on top bar
    
plt.title("Label rating is balance")
plt.show()
# replace labels with integer numbers:
dataset_train['label'] = dataset_train['label'].replace({'sad':0, 'happy':1}).astype(int)
dataset_train.head()
# read test dataset
dataset_test = pd.read_csv('EA-test.txt', header=None, names= ['text', 'emotion'] ,delimiter=';') 
dataset_test.head()
dataset_test.tail()
dataset_test.info()
dataset_test.shape
dataset_test.describe()
dataset_test.dropna(inplace=True)
dataset_test.isnull().sum()
dataset_test['emotion'].unique()
dataset_test['emotion'].value_counts()
# plot
ax= sns.countplot(x=dataset_test['emotion'],
                  data=dataset_test)

for p in ax.patches: # bars
    '''
    get_bbox(): return bounding box of the bar, 
    get_points(): returns the coordinates of the four corners of the bounding box.
    '''
    x= p.get_bbox().get_points()[:,0] # extract the x-coordinates of the four corners of the bar rectangle
    y= p.get_bbox().get_points()[1,1] # extract the y-coordinate of the top-right corner
    ax.annotate(f'{y:.0f}', (x.mean(), y), ha='center',va='bottom') # text on top bar
    
plt.title("Emotion Rating")
plt.show()
dataset_test['label'] = dataset_test['emotion']
dataset_test['label'].replace(['joy', 'love', 'surprise'], 'happy', inplace=True)
dataset_test['label'].replace(['sadness', 'anger', 'fear'], 'sad', inplace=True)
dataset_test['label'].unique()
dataset_test['label'].value_counts()
# plot
ax= sns.countplot(x=dataset_test['label'],
                  data=dataset_train)

for p in ax.patches: # bars
    '''
    get_bbox(): return bounding box of the bar, 
    get_points(): returns the coordinates of the four corners of the bounding box.
    '''
    x= p.get_bbox().get_points()[:,0] # extract the x-coordinates of the four corners of the bar rectangle
    y= p.get_bbox().get_points()[1,1] # extract the y-coordinate of the top-right corner
    ax.annotate(f'{y:.0f}', (x.mean(), y), ha='center',va='bottom') # text on top bar
    
plt.title("Label rating is balance")
plt.show()
# replace labels with integer numbers:
dataset_test['label'] = dataset_test['label'].replace({'sad':0, 'happy':1}).astype(int)
dataset_test.head()
from nltk.corpus import stopwords
stop_words= stopwords.words("english")
print(stop_words)
excluding= ['againts', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
            "didn't",'doesn', "doesn't", 'hadn', "hadn't", 'has', "hasn't", 'haven', "haven't", 'isn', 
            "isn't", 'might', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shouldn', "shouldn't", 
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stop_words= [word for word in stop_words if word not in excluding]
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
stemmer= PorterStemmer()    
lemmatizer = WordNetLemmatizer()
def preprocessing_dataset(texts):
    cleaned_texts = []  # list to include the cleaned text in.

    for sent in texts: # loop on each sentence
        filtered_sent= []
        tokens= word_tokenize(sent.lower())
        
        for token in tokens: # loop on each token from sentence
            # check if it's not numeric and its length > 2 and not in stop words
            if (not token.isnumeric()) and (len(token) > 2) and (token not in stop_words):
                filtered_sent.append(stemmer.stem(token) and lemmatizer.lemmatize(token))
                
        # convert tokens to text
        text= " ".join(filtered_sent) # string of cleaned words 
        cleaned_texts.append(text)
    
    return cleaned_texts
# cleaning the training text
dataset_train['clean_text'] = preprocessing_dataset(dataset_train['text'].values)
dataset_train.head()
## cleaning the test text
dataset_test['clean_text'] = preprocessing_dataset(dataset_test['text'].values)
# first 5
dataset_test.head()
# random 10
dataset_test.sample(10)
X_test = dataset_test['clean_text']
y_test = dataset_test['label']
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])
from sklearn.model_selection import RandomizedSearchCV
params = {
    # determines the range of n-grams to be used for tokenization.
    # (1, 2) consider unigrams and bigrams
    'tfidf__ngram_range': [(1, 1), (1, 2), (2, 2)],
    # minimum document frequency of a term in the corpus
    # increasing the value of "max_df" to exclude terms that appear too frequently in the corpus.
    'tfidf__max_df': [0.5, 0.75, 1.0],
    # maximum document frequency of a term in the corpus
    # decreasing the value of "min_df" to allow more terms to be included,
    'tfidf__min_df': [1, 2, 3],
    'nb__alpha': [0.1, 0.5, 1.0]
}
random_search = RandomizedSearchCV(pipeline,
                                   param_distributions=params,# parameters grid  
                                   n_iter=20,# number of iteration
                                   cv=5)# Cross-validation to evaluate the model's performance
random_search.fit(X_train, y_train)
pd.DataFrame(random_search.cv_results_)[["params"]]
print(f"Best parameters: {random_search.best_params_}")
print(f"Best score: {round(random_search.best_score_*100)}%")
best_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=random_search.best_params_['tfidf__ngram_range'], 
                              max_df=random_search.best_params_['tfidf__max_df'], 
                              min_df=random_search.best_params_['tfidf__min_df'])),
    ('nb', MultinomialNB(alpha=random_search.best_params_['nb__alpha']))
])

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
# compute accuracy score with y-test and y-predictions
# number of correct predictions divided by the total number of predictions.

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(accuracy*100)}%")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plt.figure(figsize = (10,10))
labels = ['sad', 'happy']
sns.heatmap(cm, 
            xticklabels=labels, 
            yticklabels=labels, 
            annot=True, 
            cmap='Blues', 
            fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
import joblib
joblib.dump(best_pipeline,'pipeline_tfidf_nb.pkl') 

# load the model use joblib.load
def preprocessing_text(text):
    cleanned_text= []
    tokens = word_tokenize(text.lower())
    
    for token in tokens: # loop on each token from sentence
        # check if it's not numeric and its length > 2 and not in stop words
        if (not token.isnumeric()) and (len(token) > 2) and (token not in stop_words):
            # make stemmer and add in filtered list
            cleanned_text.append(stemmer.stem(token) and lemmatizer.lemmatize(token))
            
    text = ' '.join(cleanned_text) 
    return text
# pre-processing
text = "I am happy"
print(preprocessing_text(text))
# Prediction
result= best_pipeline.predict([text])
print(result)
# Probability of prediction
Probability= best_pipeline.predict_proba([text])
print(Probability)
# pre-processing
text = "I don happy"
print(preprocessing_text(text))
# Prediction
result= best_pipeline.predict([text])
print(result)
# Probability of prediction
Probability= best_pipeline.predict_proba([text])
print(Probability)
import streamlit as st
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd

# Load the trained model
model = joblib.load('pipeline_tfidf_nb.pkl')

# Load stopwords and define stemmer and lemmatizer
stop_words = set(stopwords.words("english"))
excluding = ['againts', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
              "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'has', "hasn't", 'haven', "haven't", 'isn',
              "isn't", 'might', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shouldn', "shouldn't",
              'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stop_words = [word for word in stop_words if word not in excluding]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    cleaned_text = []
    tokens = word_tokenize(text.lower())
    
    for token in tokens:
        if (not token.isnumeric()) and (len(token) > 2) and (token not in stop_words):
            cleaned_text.append(stemmer.stem(token) and lemmatizer.lemmatize(token))
            
    cleaned_text = ' '.join(cleaned_text)
    return cleaned_text

# Define the main function for the Streamlit app
def main():
    st.title("Text Emotion Classifier")
    st.subheader("Enter a text and the model will predict the emotion.")
    
    text = st.text_input("Enter a text:")
    processed_text = preprocess_text(text)
    
    if st.button("Predict"):
        result = model.predict([processed_text])
        probability = model.predict_proba([processed_text])
        
        st.write("Prediction:", result[0])
        st.write("Probability:", probability[0])
    

    main()
