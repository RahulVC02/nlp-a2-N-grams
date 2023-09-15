import pandas as pd
from ngrams import LanguageModel

all_sentences = pd.read_csv('data/final_sentences_final.csv', header=None)
all_sentences = all_sentences.iloc[:,0]
split_ratio = 0.8

all_sentences = all_sentences.sample(frac=1, random_state=42)  
num_train = int(len(all_sentences) * split_ratio)

#actual train and test data
train_data = all_sentences[:num_train]
test_data = all_sentences[num_train:]

# #sanity check
# train_data = all_sentences
# test_data = train_data

print("Train Data Size: ", len(train_data))
print("Test Data Size: ", len(test_data))

#flag to find log perplexity or normal perplexity
calculate_log_perplexity=True

#possible values for smoothing - unsmoothed, laplace, add-k, goodTuring
smoothing_values = [None, 'laplace', 0.001, 0.01, 0.1, 0.5, 2, 5, 10, 'goodTuring']

#ngram models
unigram_model = LanguageModel(train_data, test_data, 1, smoothing = 'goodTuring')
print("Unigram Perplexity: ", unigram_model.get_total_perplexity(log=calculate_log_perplexity))

bigram_model = LanguageModel(train_data, test_data, 2, smoothing = 'goodTuring')
print("Bigram Perplexity: ", bigram_model.get_total_perplexity(log=calculate_log_perplexity))

trigram_model = LanguageModel(train_data, test_data, 3, smoothing = 'goodTuring')
print("Trigram Perplexity: ", trigram_model.get_total_perplexity(log=calculate_log_perplexity))

quadgram_model = LanguageModel(train_data, test_data, 4, smoothing = 'goodTuring')
print("Quadgram Perplexity: ", quadgram_model.get_total_perplexity(log=calculate_log_perplexity))