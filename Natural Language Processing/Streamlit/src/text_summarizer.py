import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import en_core_web_sm
import nltk
import heapq
import re
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

import streamlit as st
import pandas as pd

def spacy_sum(text, no_sent):

    import en_core_web_sm
    nlp = spacy.load('en_core_web_sm') # load the model (English) into spaCy

    doc = nlp(text) #pass the string doc into the nlp function.
    l = len(list(doc.sents)) #number of sentences in the given string
    st.write('Number of sentences in the original text: {}'.format(l) )

    #def generate_summary(text, no_sent):

    doc = nlp(text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')
        
    ## Word frequency 
    word_frequencies = {}  
    for word in nltk.word_tokenize(text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    ## Sentence scoring based on word frequency
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(int(no_sent), sentence_scores, key=sentence_scores.get)
    summary = ''.join(summary_sentences)
    
    return summary

#lsa
def lsa_sum(text, no_sent):

    LANGUAGE = "english"
    SENTENCES_COUNT = int(no_sent)
    summary = []

    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

    ss = []

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
            ss.append(str(sentence))

    summary = ''.join(ss)
    return summary

#luhn
def luhn_sum(text, no_sent):

    parser=PlaintextParser.from_string(text,Tokenizer('english'))

    SENTENCES_COUNT = int(no_sent)

    #  Creating the summarizer
    luhn_summarizer=LuhnSummarizer()
    luhn_summary=luhn_summarizer(parser.document,sentences_count=SENTENCES_COUNT)

    ss = []

    # Printing the summary
    for sentence in luhn_summary:
          ss.append(str(sentence))
            
    summary = ''.join(ss)
    return summary

#kl
def kl_sum(text, no_sent): 
    
    parser=PlaintextParser.from_string(text,Tokenizer('english'))

    SENTENCES_COUNT = int(no_sent)

    # Instantiating the  KLSummarizer
    kl_summarizer=KLSummarizer()
    kl_summary=kl_summarizer(parser.document,sentences_count=SENTENCES_COUNT)

    ss = []

    # Printing the summary
    for sentence in kl_summary:
        ss.append(str(sentence))

        summary = ''.join(ss)
        return summary

#t5
def t5_abs(text):

	# Importing requirements
	from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

	# Instantiating the model and tokenizer 
	my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
	tokenizer = T5Tokenizer.from_pretrained('t5-small')

	# encoding the input text
	input_ids=tokenizer.encode(text, return_tensors='pt', max_length=512)

	# Generating summary ids
	summary_ids = my_model.generate(input_ids)

	# Decoding the tensor and printing the summary.
	t5_summary = tokenizer.decode(summary_ids[0])
	return (t5_summary)

#gpt2 abstractive
def gpt2_abs(text):

	# Importing model and tokenizer
	from transformers import GPT2Tokenizer,GPT2LMHeadModel

	# Instantiating the model and tokenizer with gpt-2
	tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
	model=GPT2LMHeadModel.from_pretrained('gpt2')

	# Encoding text to get input ids & pass them to model.generate()
	inputs=tokenizer.batch_encode_plus([text],return_tensors='pt',max_length=512)
	summary_ids=model.generate(inputs['input_ids'],early_stopping=True)

	# Decoding the tensor and printing the summary.
	GPT_summary=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
	return (GPT_summary)

#bart abtractive
def bart_abs(text):

	# Importing the model
	from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

	# Loading the model and tokenizer for bart-large-cnn

	tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
	model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

	# Encoding the inputs and passing them to model.generate()
	inputs = tokenizer.batch_encode_plus([text],return_tensors='pt')
	summary_ids = model.generate(inputs['input_ids'], early_stopping=True)

	# Decoding and printing the summary
	bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
	return (bart_summary)

#xlm abtractive
def xlm_abs(text):

	# Importing model and tokenizer
	from transformers import XLMWithLMHeadModel, XLMTokenizer

	# Instantiating the model and tokenizer 
	tokenizer=XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
	model=XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')

	# Encoding text to get input ids & pass them to model.generate()
	inputs=tokenizer.batch_encode_plus([text],return_tensors='pt',max_length=512)
	summary_ids=model.generate(inputs['input_ids'],early_stopping=True)

	# Decode and print the summary
	XLM_summary=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
	return (XLM_summary)
