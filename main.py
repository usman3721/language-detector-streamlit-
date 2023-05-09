import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import re



cv=CountVectorizer()
st.set_option("deprecation.showfileUploaderEncoding",False)

df1=pd.read_csv("./dataset.csv")
data=df1.copy()

def data_cleaner(Text):
    Text=re.sub(r'[\/`!@#$%^&*()_+{}<>,.?/":;0-9]',' ',Text)
    Text=Text.lower()
    return Text


data["cleaned_data"]=""

data["cleaned_data"]=data["Text"].apply(lambda x:data_cleaner(x))
data.drop("Text",axis=1,inplace=True)

x=np.array(data["cleaned_data"],)
y=np.array(data["language"])





X=cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)


le=LabelEncoder()
y=le.fit_transform(y)

model=MultinomialNB()
model.fit(X_train,y_train)


def main():
    st.sidebar.header("lanuage Detection")
    st.sidebar.text("This is a web app that tell contain 20 language trained with a model,i.e the app can different 20 languages")
    st.sidebar.header("just fill in the information below")
    st.sidebar.text("naice bayes model was used")
pred_review_text=st.text_input("Enterr the laguage")
if st.button("predict"):
    updated_text=cv.transform([pred_review_text]).toarray()
    result=model.predict(updated_text)
    st.write(result)


