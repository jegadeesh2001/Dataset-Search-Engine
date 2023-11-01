<div align="center">
<h1> INFORMATION RETRIEVAL PACKAGE
</h1>



</div>




![chrome-capture-2023-10-1 (1)](https://github.com/jegadeesh2001/Dataset-Search-Engine/assets/62760269/6e3351e5-3f3d-4fb8-841e-a7a8380224f1)


## Objective

The project aims to build an interactive search engine that helps users to find relevant datasets from a corpus of datasets that satisfies certain keyword conditions specified by the users

## Architecture Diagram

![Screenshot 2023-11-01 181416](https://github.com/jegadeesh2001/Dataset-Search-Engine/assets/62760269/cede37b4-b670-4b70-a2b8-d26cfba7cc23)


## Detailed Approach Notes and Future Works
The Approach and the workflow of the project are provided in the slides

https://docs.google.com/presentation/d/1H2twHuA146wSInGZu8ExNs_whIuFVPJDPPiJkrR_sA0/edit?usp=sharing

## Repository Structure
1. Frontend - Consists of the files used to develop web application using Streamlit and the corresponding documentation
   https://github.com/jegadeesh2001/Dataset-Search-Engine/tree/main/FrontEnd
2. Colab_Notebooks - Consists of all the main workflow files and the files containing various approaches before the optimal model.
   https://github.com/jegadeesh2001/Dataset-Search-Engine/tree/main/Colab%20Files

## Technologies Used

  1. Beautiful Soup - For scraping Datasets from Google Dataset Search.
  
  2. NLTK - For Preprocessing and cleaning the scraped data.
  
  3. SKLearn - For calculating Similarity Measures, Tfidf, and bag of words.
  
  4. BM25 - Used for building keyword-based probabilistic search engine.
  
  5. Hugging Face - Used for performing embedding using sentence transformers in semantic search engine
  
  6. FAISS - Used for calculating the similarity between document embedding and query embedding in semantic search engine.
  
  7. Plotly - For plotting interactive charts.
  
  8. Streamlit - For building the web application where the retrieval models were deployed











