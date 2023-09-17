import pandas as pd
from ngrams import LanguageModel
from matplotlib import pyplot as plt

all_sentences = pd.read_csv('data/final_sentences_final.csv', header=None)
all_sentences = all_sentences.iloc[:,0]

#sanity check
# train_data = all_sentences
# test_data = train_data

split_ratio = 0.8
all_sentences = all_sentences.sample(frac=1, random_state=42)  
num_train = int(len(all_sentences) * split_ratio)

#actual train and test data
train_data = all_sentences[:num_train]
test_data = all_sentences[num_train:]
# test_data = all_sentences


print("No. of Sentences in Train Data : ", len(train_data))
print("No. of Sentences in Test Data : ", len(test_data))

#possible values for smoothing - unsmoothed, laplace, add-k, goodTuring
smoothing_values = [None, 'laplace', 0.001, 0.01, 0.1, 0.5, 2, 5, 10, 'goodTuring']

#flag to find log perplexity or normal perplexity
calculate_log_perplexity = True
# for current_smoothing in smoothing_values[2:9]:
#     # current_smoothing = "goodTuring"
#     print("Smoothing : ", current_smoothing)

#     arr = []
#     unigram_model = LanguageModel(train_data, test_data, 1, smoothing = current_smoothing)
#     arr.append(unigram_model.get_total_perplexity(log=calculate_log_perplexity))

#     bigram_model = LanguageModel(train_data, test_data, 2, smoothing = current_smoothing)
#     arr.append(bigram_model.get_total_perplexity(log=calculate_log_perplexity))

#     trigram_model = LanguageModel(train_data, test_data, 3, smoothing = current_smoothing)
#     arr.append(trigram_model.get_total_perplexity(log=calculate_log_perplexity))

#     quadgram_model = LanguageModel(train_data, test_data, 4, smoothing = current_smoothing)
#     arr.append(quadgram_model.get_total_perplexity(log=calculate_log_perplexity))

#     pentagram_model = LanguageModel(train_data, test_data, 5, smoothing = current_smoothing)
#     arr.append(pentagram_model.get_total_perplexity(log=calculate_log_perplexity))

#     hexagram_model = LanguageModel(train_data, test_data, 6, smoothing = current_smoothing)
#     arr.append(hexagram_model.get_total_perplexity(log=calculate_log_perplexity))

#     septagram_model = LanguageModel(train_data, test_data, 7, smoothing = current_smoothing)
#     arr.append(septagram_model.get_total_perplexity(log=calculate_log_perplexity))

#     plt.plot(arr, label="k = {}".format(current_smoothing))

current_smoothing = "goodTuring with conditional"
print("Smoothing : ", current_smoothing)

arr = []
unigram_model = LanguageModel(train_data, test_data, 1, smoothing = current_smoothing)
arr.append(unigram_model.get_total_perplexity(log=calculate_log_perplexity))

bigram_model = LanguageModel(train_data, test_data, 2, smoothing = current_smoothing)
arr.append(bigram_model.get_total_perplexity(log=calculate_log_perplexity))

trigram_model = LanguageModel(train_data, test_data, 3, smoothing = current_smoothing)
arr.append(trigram_model.get_total_perplexity(log=calculate_log_perplexity))

quadgram_model = LanguageModel(train_data, test_data, 4, smoothing = current_smoothing)
arr.append(quadgram_model.get_total_perplexity(log=calculate_log_perplexity))

pentagram_model = LanguageModel(train_data, test_data, 5, smoothing = current_smoothing)
arr.append(pentagram_model.get_total_perplexity(log=calculate_log_perplexity))

hexagram_model = LanguageModel(train_data, test_data, 6, smoothing = current_smoothing)
arr.append(hexagram_model.get_total_perplexity(log=calculate_log_perplexity))

septagram_model = LanguageModel(train_data, test_data, 7, smoothing = current_smoothing)
arr.append(septagram_model.get_total_perplexity(log=calculate_log_perplexity))

plt.plot(arr)


plt.legend()
plt.xlabel("N")
plt.ylabel("log Perplexity")
plt.title("Good Turing with conditional and Interpolated counts")
plt.show()

