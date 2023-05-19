
# Importing of Libaries
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle


pickled_model = pickle.load(open('detector.model', 'rb'))
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
label_encoder = pickle.load(open('label_encoder', 'rb'))


# Creating a function to be used in streamlit
def main():
    st.sidebar.header("Language Detector")
    st.sidebar.text("This is a web app that tell contain 20 language trained with a model,i.e the app can different 20 languages")
    st.sidebar.header("just fill in the information below")
    st.sidebar.text("Naive Bayes model was used")
pred_review_text=st.text_input("Enter a sentence in a particular language")

# A conditional statement to display the result using Streamlit
if st.button("Detect"):
    lang=pickled_model.predict(loaded_vectorizer.transform([pred_review_text]))
    lang=label_encoder.inverse_transform(lang)
    st.write(lang[0])


# This is a lamguage detector 