import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import re
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('punkt')
st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")



st.markdown(
    """
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #008B8B;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




st.title("Exploratory Data Analysis")



def preprocess(data):
  data=data.lower()
  data=data.replace("\n"," ")
  data=re.sub("[^a-zA-Z ]+","",data)
  words = data.split()
  stop_words = set(stopwords.words('english'))
  words = [word for word in words if word not in stop_words]

  processed_data = " ".join(words)
  return processed_data




df = pd.read_csv('dataset_meta.csv')
df['description']=df['description'].apply(preprocess)

with st.container():
    st.markdown("""---""")
    st.title("Length Distribution")
    df['text_length'] = df['description'].apply(len)

    fig = px.histogram(df, x='text_length', title='Text Length Distribution')
    st.plotly_chart(fig,use_container_width=True)

with st.container():
    st.markdown("""---""")
    st.title("Word Frequency Distribution")

    df['tokens'] = df['description'].apply(nltk.word_tokenize)

    # Calculate word frequencies
    all_words = [word for tokens in df['tokens'] for word in tokens]
    word_freq = FreqDist(all_words)

    # Create a bar chart for the most common words
    common_words = word_freq.most_common(20)
    common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

    fig = px.bar(common_words_df, x='Word', y='Frequency', title='Most Common Words')
    st.plotly_chart(fig,use_container_width=True)

with st.container():
    st.markdown("""---""")
    st.title("Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    fig = go.Figure(data=[go.Image(z=wordcloud)])
    fig.update_layout(title='Word Cloud of Most Common Words')
    st.plotly_chart(fig,use_container_width=True)


with st.container():
    st.markdown("""---""")
    st.title("Zipf's Law")
    word_freq = FreqDist(all_words)
    common_words = word_freq.most_common(20)
    common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])

    # Create Zipf's Law plot
    rank = common_words_df.index + 1
    frequency = common_words_df['Frequency']

    fig = px.scatter(x=rank, y=frequency, title="Zipf's Law")
    fig.update_layout(xaxis_type='log', yaxis_type='log')
    fig.update_xaxes(title='Rank (log scale)')
    fig.update_yaxes(title='Frequency (log scale)')
    st.plotly_chart(fig,use_container_width=True)


with st.container():
    st.markdown("""---""")
    st.title("Heap's Law")
    vocabulary_size = []
    num_tokens = []

    for i, tokens in enumerate(df['tokens']):
        num_tokens.append(len(tokens))
        unique_words = set(tokens)
        vocabulary_size.append(len(unique_words))

    # Create Heaps' Law plot
    fig = px.line(x=num_tokens, y=vocabulary_size, title="Heaps' Law")
    fig.update_xaxes(title='Number of Tokens')
    fig.update_yaxes(title='Vocabulary Size')
    st.plotly_chart(fig,use_container_width=True)
