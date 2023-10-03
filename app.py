import streamlit as st
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer()

nltk.download('stopwords')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    l = []
    for word in text:
        if word.isalnum():
            if word not in stopwords.words('english') and word not in string.punctuation:
                word = ps.stem(word)
                l.append(word)

    return " ".join(l)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load((open('model.pkl', 'rb')))
st.title("SMS/Email Spam Classifier")

input_msg = st.text_input("Enter the message : ")

if st.button('Predict'):
    # preprocess the msg
    transformed_msg = transform_text(input_msg)
    # vectorize the msg
    vector_input = tfidf.transform([transformed_msg])
    # predict
    result = model.predict(vector_input)[0]
    # Display
    if result == 0:
        st.header('Not Spam')
    else:
        st.header("Spam")