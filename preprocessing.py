import pandas as pd
import re
import pandas as pd
import nltk
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
    corpus += comment + ' '

corpus_comments = []
sentences = sent_tokenize(corpus)
for sentence in sentences:
    sentence = re.sub(r'[^a-zA-Z0-9 ]', '', sentence)   #remove punctuation after tokenizing
    corpus_comments.append(sentence)

pd.Series(corpus_comments).to_csv('data/comments.csv', index=False, header=False)
final_df = pd.read_csv('data/comments.csv', header=None)

print(len(final_df))
print(final_df[:10])
