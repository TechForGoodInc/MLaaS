import streamlit as st
import pandas as pd
import base64
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


def input():
    no_kw = st.number_input("How many Keywords do you want?")
    n_gram = st.number_input("Enter the n-gram")
    raw_text = st.text_area("Your Text")
    return no_kw, n_gram, raw_text

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def sent_trans(raw_text, n_gram, no_kw, model_name): 
    
    model_ = SentenceTransformer(model_name)
    kw_model = KeyBERT(model=model_)


    kw = []
    weight = []

    a = kw_model.extract_keywords(raw_text, keyphrase_ngram_range=(int(n_gram), int(n_gram)), stop_words='english', 
                              use_maxsum=True, nr_candidates=int(no_kw), top_n=int(no_kw))

    for i in list(range(len(a))):
            kw.append(a[i][0])
            weight.append(a[i][1])
            

    df = pd.DataFrame()    
    df['Keyword'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df  

def write_df(df):
    st.write(df)
    tmp_download_link = download_link(df, 'keywords.csv', 'Download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)



def Sentence_Transformer():

    st.subheader("Sentence Transformer")

    options = ['paraphrase-mpnet-base-v2', "paraphrase-MiniLM-L6-v2", "paraphrase-multilingual-mpnet-base-v2", "paraphrase-TinyBERT-L6-v2", "paraphrase-distilroberta-base-v2"]

    st_mod = st.selectbox("Select a Tansformer Model", options) 

    if st_mod == "paraphrase-mpnet-base-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords", key = "321"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-mpnet-base-v2")
                    write_df(df)

    if st_mod == "paraphrase-MiniLM-L6-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-MiniLM-L6-v2")
                    write_df(df)

    if st_mod == "paraphrase-multilingual-mpnet-base-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-multilingual-mpnet-base-v2")
                    write_df(df)

    if st_mod == "paraphrase-TinyBERT-L6-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-TinyBERT-L6-v2")
                    write_df(df)

    if st_mod == "paraphrase-distilroberta-base-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-distilroberta-base-v2")
                    write_df(df)

    
# ------------------------------------------------------------------------------------------------------------------------- Sentence Transformer ends

