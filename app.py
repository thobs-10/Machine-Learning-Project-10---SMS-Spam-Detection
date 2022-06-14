import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize

ps = PorterStemmer()

def transform_text(text):
  #lowercase
  text = text.lower()
  #tokenization
  text = word_tokenize(text)
  # remove special characters
  y=[] # list that will contain are alphanumeric(alphabetic and numerical)
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:] # take everything from start to last one, place in in text
  y.clear() # remove alphanumeric

  # remove stop words and punctuation
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  # apply porter stemming
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('spam_clf_model.pkl','rb'))

st.title("Email\SMS Spam Classifier")

# text area for text input
input_text = st.text_area("Enter\paste the message or email")

if st.button('Predict'):

  # 1. preprocess
  transformed_input = transform_text(input_text)
  # 2. vectorize
  vector_input = tfid.transform([transformed_input])
  # 3. predict
  results = model.predict(vector_input)[0]
  # 4. Display
  if results == 1:
    st.header("Spam")
  else:
    st.header("Not Spam")
