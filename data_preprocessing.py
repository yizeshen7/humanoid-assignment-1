''' Filename:   data_preprocessing.py
    Author:     Aditti Ramsisaria
    Brief:      This file uses libraries like nltk (and maybe spaCy) to preprocess
                news articles csv file for natural language processing
'''

import pandas as pd
import string
import re
import nltk
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
df = pd.read_csv("bbc_dataset.csv")

# remove duplicate data
data = df.drop_duplicates(subset = ["title", "description", "category"], keep = "first", inplace = False)
# keep only description and category of article
data = data[["title", "description", "category"]]
# remove incomplete entries
data = data.dropna()

# lowercasing, noise removal, stopword removal
def data_cleanup(text):
    cleaned_text = text.lower()
    # remove html markup
    cleaned_text = re.sub("(<.*?>)", "", cleaned_text)
    # remove punctuation
    cleaned_text = re.sub("[%s]" % re.escape(string.punctuation), "", cleaned_text)
    cleaned_text = re.sub("[‘’“”…]", "", cleaned_text)
    cleaned_text = re.sub("\n", "", cleaned_text)
    # remove stopwords
    clean_list = [w for w in cleaned_text if w not in stop_words]
    cleaned_text = " ".join(clean_list)
    return cleaned_text

# tokenization, lemmitization
def normalization(text):
    token_list = word_tokenize(text)
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in token_list]
    normalised_text = " ".join(lemma_list)
    return normalised_text

data["preprocessed"] = data["description"].apply(func = data_cleanup)
data["preprocessed"] = data["description"].apply(func = normalization)

# POS tagging 
data["pos_tags"] = pos_tag_sents(data["preprocessed"].apply(word_tokenize).tolist())

data.to_csv("./tagged_data.csv")


