import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
from rank_bm25 import *
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist


df=pd.read_csv('dataset_meta.csv')
corpus=df['description'].tolist()

tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def search(query):
    query=query.lower()
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    
    index=np.argsort(-doc_scores)[:5]
    top_result_row = df.iloc[index]

    return top_result_row

st.markdown("<h1 style='text-align: center;'>KEYWORD SEARCH ENGINE 🔎</h1>", unsafe_allow_html=True)
st.text(" ")
q=st.text_input("Enter the Query for Searching")
st.text(" ")

    
if q!='':
        if st.button("Search"):
            episode=search(str(q))
            with st.container():
                st.markdown("""---""")
                st.title("The Retrieved DataSet")
                st.write(episode)
                st.markdown("""---""")
                for _, row in episode.iterrows():
                        
                        
                        st.write(f"NAME: {row['name']}")
                        st.write(f"DESCRIPTION: {row.description}")
                        st.write(f"URL: {row.url}")
                        st.markdown("""---""")

            with st.container():
                 st.markdown("""---""")
                 st.title("Visualization of Words in Extracted Datasets")
                 df=episode.copy()
                 
                 df['tokens'] = df['description'].apply(nltk.word_tokenize)


                 all_words = [word for tokens in df['tokens'] for word in tokens]
                 word_freq = FreqDist(all_words)

                # Create a bar chart for the most common words
                 common_words = word_freq.most_common(20)
                 common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

                 fig = px.bar(common_words_df, x='Word', y='Frequency', title='Most Common Words')
                 st.plotly_chart(fig,use_container_width=True)