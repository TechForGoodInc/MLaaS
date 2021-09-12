import streamlit as st
import numpy as np
import sys
from src.keyword_ex import *
from src.text_summarizer import *
from src.sentiment import *
from src.pos_ner import *
import en_core_web_sm


if __name__ == '__main__':


    st.markdown("<h1 style='text-align: center;'>Natural Language Processing</h1>", unsafe_allow_html=True)
    st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

    #st.title("Natural Language Processing")

    menu = ["Keyword Extraction", "Information Extraction", "Text Summarization", "Sentiment Analysis"]
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

    if choice == "Information Extraction":
        posner()

    if choice == "Text Summarization":

        st.markdown("<h3 style='text-align: center;'>Text Summarization</h3>", unsafe_allow_html=True)

        menu_main = ["Extractive Text Summarization", "Abstractive Text Summarization"]
        choice_main = st.selectbox("Select", menu_main)

        if choice_main == "Extractive Text Summarization":
            type_task = st.radio("Select from below", ("spaCy", "Latent Semantic Analysis", "Luhn", "KL-Sum"))

            if type_task == 'spaCy':
                    st.subheader("spaCy")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = spacy_sum(raw_text, no_sent)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_spacy.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)


            if type_task == 'Latent Semantic Analysis':
                    st.subheader("Sumy: Latent Semantic Analysis")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = lsa_sum(raw_text, no_sent)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_LSA.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

            if type_task == 'Luhn':
                    st.subheader("Sumy: Luhn")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = luhn_sum(raw_text, no_sent)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_Luhn.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

            if type_task == 'KL-Sum':
                    st.subheader("Sumy: KL-Sum")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = kl_sum(raw_text, no_sent)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_KL-Sum.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

        
        if choice_main == "Abstractive Text Summarization":
            type_task = st.radio("Select Transformer Model", ("BART", "XLM", "T5", "GPT-2"))

            if type_task == 'T5':
                    st.subheader("T5")

                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = t5_abs(raw_text)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_T5.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)


            if type_task == 'BART':
                    st.subheader("BART")

                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = bart_abs(raw_text)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_bart.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)


            if type_task == 'XLM':
                    st.subheader("XLM")

                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = xlm_abs(raw_text)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_xlm.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)


            if type_task == 'GPT-2':
                    st.subheader("GPT-2")

                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = gpt2_abs(raw_text)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_gpt2.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

    if choice == "Sentiment Analysis":
        sent() 