import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from rank_bm25 import *

df=pd.read_csv('dataset_meta.csv')
corpus=df['description'].tolist()

tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def search(query):
    query=query.lower()
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    index=np.argsort(-doc_scores)[:5][0]
    return df.iloc[index][0]

st.title("Search Engine using Okapi")

q=st.text_input("Enter the Query for Searching")

with st.container():
    st.markdown("""---""")
    
    if q is not None:
        episode=search(str(q))
        st.title("The Retrieved DataSet")
        st.write(episode)