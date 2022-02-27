# MOVIE RECOMMENDATION SYSTEM USING MACHINE LEARNING

# Importing Libraries

# Python pillow library is used to image class within it to show the image
from PIL import Image
import numpy as np
import pandas as pd
# Streamlit is an open source app framework in Python language. It helps us create web apps for data science and machine learning in a short time.
import streamlit as st
# Difflib is a Python module that contains several easy-to-use functions and classes that allow users to compare sets of data.
import difflib
# It is a simple technique to vectorize text documents — i.e. transform sentences into arrays of numbers
from sklearn.feature_extraction.text import TfidfVectorizer
# In the sklearn module, there is an in-built function called cosine_similarity() to calculate the cosine similarity.
from sklearn.metrics.pairwise import cosine_similarity

# Dataset

# Loading the dataset - "movies.csv" to "df" using pandas
df = pd.read_csv('movies.csv')

# Dataset preprocessing - Selecting required features or datas for the similarity
sf = ['genres', 'keywords', 'tagline', 'cast', 'director']
# The given dataset contains null values in some fields, so preprocessing the dataset is required to fill null values.
for i in sf:
    df[i] = df[i].fillna('')

# Merging all the features to a single one for Vectorization
mf = df['genres']+''+df['keywords']+'' + \
    df['tagline']+''+df['cast']+''+df['director']
# TfidfVectorizer - Transforms text to feature vectors that can be used as input to estimator.
vect = TfidfVectorizer()
fv = vect.fit_transform(mf)

# Cosine Similarity

# Cosine similarity measures the similarity between two vectors of an inner product space. Cosine similarity is one of the metric to measure the text-similarity
similarity = cosine_similarity(fv)

# Streamlit

# Pillow is a Python Imaging Library (PIL), which adds support for opening, manipulating, and saving images.
img = Image.open("mv.jpg")
# Streamlit - Display an image or list of images.
st.image(img, width=700)

# Streamlit - Display text in title formatting
st.title('MOVIE RECOMMENDATION SYSTEM USING MACHINE LEARNING')
# Streamlit - Write arguments to the app.
st.write("ENTER YOUR FAVOURITE MOVIE : ")
# Streamlit - Display a single-line text input widget.
movie_name = st.text_input('Enter the Movie name : ')
# Streamlit - Display a button widget
submit = st.button("SUBMIT")

# Output

# The try block lets you test a block of code for errors.
try:
    if submit:
        # mlist stores the list of all movie titles from the movies.csv dataset.
        mlist = df['title'].tolist()
        # get_close_matches() is a function that is available in the difflib Python package.
        # The difflib module provides classes and functions for comparing sequences.
        # It finds the close matches of movies in mlist from the USER INPUT.
        match = difflib.get_close_matches(movie_name, mlist)
        # Finds the index of closely matched movie
        index = df[df.title == match[0]]['index'].values[0]
        # Finds the similarity of the user input with each movie in the dataset like a loop.
        ss = list(enumerate(similarity[index]))
        # Sorting the similar values is descending or reverse order, so that the more similar movies or values comes first.
        sortedlist = sorted(ss, key=lambda x: x[1], reverse=True)
        # Writing or Printing the TOP 10, most similar movies in the sorted list using a for loop.
        st.write("YOU MAY ALSO LIKE : ")
        k = 1
        for j in sortedlist:
            ind = j[0]
            x = df[df.index == ind]['title'].values[0]
            if(k <= 10):
                st.write(k, ".", x)
                k += 1
# The except block lets you handle the error.
except:
    st.write("MOVIE NOT FOUND")
    st.write("PLEASE ENTER THE CORRECT MOVIE NAME")
