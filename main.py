
# Importing of Libaries
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import re



# To update the configuration  values
st.set_option("deprecation.showfileUploaderEncoding",False)

# Reading the data
df1=pd.read_csv("./dataset.csv")
data=df1.copy()

# Create a function to clean my data
def data_cleaner(Text):
    Text=re.sub(r'[\/`!@#$%^&*()_+{}<>,.?/":;0-9]',' ',Text)
    Text=Text.lower()
    return Text

# Cleaning of data and creating a column for the cleaned data
data["cleaned_data"]=""
data["cleaned_data"]=data["Text"].apply(lambda x:data_cleaner(x))

# Dropping the formal column that contain unclean data
data.drop("Text",axis=1,inplace=True)

# Turning columns to array
x=np.array(data["cleaned_data"])
y=np.array(data["language"])

# Normalizing the data
le=LabelEncoder()
y=le.fit_transform(y)

# Tranforming of text to vector form
cv=CountVectorizer()
X=cv.fit_transform(x)

# Splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)

# Creating an object for the algorithm
model=MultinomialNB()

# Fitting of the model
model.fit(X_train,y_train)


# Creating a function to be used in streamlit
def main():
    st.sidebar.header("Language Detector")
    st.sidebar.text("This is a web app that tell contain 20 language trained with a model,i.e the app can different 20 languages")
    st.sidebar.header("just fill in the information below")
    st.sidebar.text("Naive Bayes model was used")
pred_review_text=st.text_input("Enter a sentence in a particular language")

# A conditional statement to display the result using Streamlit
if st.button("Detect"):
    x=cv.transform([pred_review_text]).toarray()
    lang=model.predict(x)
    lang=le.inverse_transform(lang)
    st.write(lang[0])


# This is a lamguage detector 