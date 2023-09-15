import re
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from nltk.tokenize import sent_tokenize

df = pd.read_csv('data/comments_unfiltered.csv')

def pre_process(text):
    if text == '[removed]' or text == '[deleted]':
        text = ''
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9.,?! ]', '', text)
    return text

processed_comments = Parallel(n_jobs=8)(delayed(pre_process)(text) for text in df['Comment'])

#concatenate all comments into one string
corpus = ""
for comment in processed_comments:
    corpus += comment + '. '

def remove_punct_post_tokenizing(sentence):
    sentence = re.sub(r'[^a-zA-Z0-9 ]', '', sentence)
    sentence = re.sub(r'x000d', ' ', sentence)      #remove common usernames
    sentence = re.sub(r'x200b', ' ', sentence)      #remove common usernames
    sentence = re.sub(r' +', ' ', sentence)              #remove extra spaces
    return sentence

sentences = sent_tokenize(corpus)
corpus_comments = Parallel(n_jobs=8)(delayed(remove_punct_post_tokenizing)(sentence) for sentence in sentences)
sanitized_corpus_comments = np.array(corpus_comments)
sanitized_corpus_comments = sanitized_corpus_comments[sanitized_corpus_comments != '']

final_sent_df = pd.Series(sanitized_corpus_comments)
final_sent_df = final_sent_df.dropna()

final_sent_df.to_csv('data/final_sentences_final.csv', index=False, header=False)
final_sent_df = pd.read_csv('data/final_sentences_final.csv', header=None)

print(len(final_sent_df))
print(final_sent_df[:10])