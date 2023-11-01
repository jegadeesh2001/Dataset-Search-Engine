import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import re
from transformers import AutoTokenizer, TFAutoModel
from datasets import load_from_disk

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

embeddings_dataset = load_from_disk("Embedded_data")
embeddings_dataset.add_faiss_index(column="embeddings")


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def retrieve_data(query):
    question_embedding = get_embeddings([query]).numpy()
    scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
    )
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)
    return samples_df


st.markdown("<h1 style='text-align: center;'>SEMANTIC SEARCH ENGINE 🔎</h1>", unsafe_allow_html=True)

st.text(" ")
q=st.text_input("Enter the Query for Searching")
st.text(" ")

    
if q is not "":
    if st.button("Search"):
            episode=retrieve_data(str(q))
            with st.container():
                
                st.title("The Retrieved DataSet")
                st.write(episode)
                st.markdown("""---""")
                for _, row in episode.iterrows():
                    
                    
                    st.write(f"NAME: {row['name']}")
                    st.write(f"DESCRIPTION: {row.description}")
                    st.write(f"URL: {row.url}")
                    st.write(f"SCORE: {np.round(row['scores'],2)}")
                    st.markdown("""---""")

            with st.container():
                
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

            