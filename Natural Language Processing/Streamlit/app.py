import streamlit as st
from src.keyword_ex import *

if __name__ == '__main__':

    st.markdown("<h1 style='text-align: center;'>Natural Language Processing</h1>", unsafe_allow_html=True)

    #st.title("Natural Language Processing")

    menu = ["Keyword Extraction"]
    choice = st.sidebar.selectbox("Natural Language Processing", menu)

    if choice == "Keyword Extraction":

        st.markdown("<h3 style='text-align: center;'>Unsupervised Keyword Extraction</h3>", unsafe_allow_html=True)

        menu_main = ["keyBERT", "PKE", "RAKE", "YAKE"]
        choice_main = st.selectbox("Select", menu_main)

        if choice_main == "keyBERT":
            type_task = st.radio("Select an Embedding Model", ("Sentence Transformer", "Flair"))
            if type_task == 'Sentence Transformer':
                Sentence_Transformer()
            if type_task == 'Flair': 
                Flair()

        if choice_main == "PKE":
            PKE()
        if choice_main == "RAKE":
            rake_()
        if choice_main == "YAKE":
            yake_()