import pandas as pd
from ngrams import LanguageModel

all_sentences = pd.read_csv('data/final_sentences_final.csv', header=None)
all_sentences = all_sentences.iloc[:,0]
split_ratio = 0.8

# all_comments = all_comments.sample(frac=1, random_state=42)  
num_train = int(len(all_sentences) * split_ratio)
train_data = all_sentences[:num_train]
test_data = all_sentences[num_train:]

print("Train Data Size: ", len(train_data))
print("Test Data Size: ", len(test_data))

unigram_model = LanguageModel(train_data, test_data, 1)
print("Unigram Perplexity: ", unigram_model.get_total_perplexity())

bigram_model = LanguageModel(train_data, test_data, 2)
print("Bigram Perplexity: ", bigram_model.get_total_perplexity())

trigram_model = LanguageModel(train_data, test_data, 3)
print("Trigram Perplexity: ", trigram_model.get_total_perplexity())

quadgram_model = LanguageModel(train_data, test_data, 4)
print("Quadgram Perplexity: ", quadgram_model.get_total_perplexity())