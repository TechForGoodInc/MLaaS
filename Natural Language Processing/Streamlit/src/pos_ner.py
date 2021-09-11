import streamlit as st
import pandas as pd
import base64
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.chunk import RegexpParser

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def nltk(df, col_name):

	df.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
	df = df[[col_name]]
	df[col_name] = df[col_name].astype('str')
	df['POS'] = ''
	pattern = 'NP: {<DT>?<JJ>*<NN>}'

	def preprocess(sent):
	    sent = word_tokenize(sent)
	    sent = pos_tag(sent)
	    return sent

	for i in range(len(df)):
		df['POS'][i] = preprocess(df[col_name][i])
		cp = RegexpParser(pattern)
		cs = cp.parse(df['POS'][i])
		df['POS'][i] = tree2conlltags(cs)

	df.POS = df.POS.astype(str)
	df.columns = [col_name,'(Word, POS, NER)']
	return df

def posner():

	#st.subheader("POS Tagging & Named Entity Recognition")
	st.markdown("<h3 style='text-align: center;'>POS Tagging & Named Entity Recognition</h3>", unsafe_allow_html=True)
	uploaded_file = st.file_uploader("Choose a file")

	while True:
		try:

			df = pd.read_csv(uploaded_file)
			df1 = df.copy()
			st.subheader("First Five Rows of the Input DatFrame")
			st.write(df.head())
	                
			col_name = st.text_input("Enter the name of the column with text data.")
			col_name = str(col_name)

			try:

				df1 = df1[[col_name]]
				df1[col_name] = df1[col_name].replace('', np.nan)
				df1 = df1.dropna()
				st.subheader("Column with Text Data")
				st.write(df1)

				type_task = st.radio("Select from below", ("NLTK", "spaCy"))

				if type_task ==  "NLTK": 

							st.subheader("NLTK: Noun Phrase Chunking")
							st.markdown('Our chunk pattern consists of one rule, that a noun phrase, NP, should be formed whenever the chunker finds an optional determiner, DT, followed by any number of adjectives, JJ, and then a noun, NN.')
							st.info("pattern = 'NP: {<DT>?<JJ>*<NN>}'")
							df_nltk = nltk(df, col_name)

							st.subheader("Resulting DataFrame")
							st.write(df_nltk)
							tmp_download_link = download_link(df_nltk, 'nltk_pos_ner.csv', 'Download as CSV')
							st.markdown(tmp_download_link, unsafe_allow_html=True)

			except KeyError:        
				st.warning('Please input the column name (case-sensitive).')        
				break
                    
			break
		except ValueError:
			st.warning('Please upload the csv file to proceed.')
			break