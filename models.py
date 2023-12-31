import pandas as pd
from prettytable import PrettyTable
from ngrams import LanguageModel

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

print("No. of Sentences in Train Data : ", len(train_data))
print("No. of Sentences in Test Data : ", len(test_data))

#possible values for smoothing - unsmoothed, laplace, add-k, goodTuring
smoothing_values = [None, 'laplace', 0.001, 0.1, 0.5, 5, 'goodTuring', 'goodTuring with conditional']

#flag to find log perplexity or normal perplexity
calculate_log_perplexity = True

# initialise a prettytable and set the column headers
comparsionTable = PrettyTable(["Smoothing Technique", "", "Unigram", "Bigram", "Trigram", "Quadgram", "Pentagram"])

for curr_smoothing in smoothing_values:

    if type(curr_smoothing) in (int,float):
        smoothing_outcomes = [f"Additive : k={curr_smoothing}", ""]
    else:   
        smoothing_outcomes = [curr_smoothing, ""]

    #ngram models
    for n in range(1,6):
        ngram_model = LanguageModel(train_data, test_data, n, smoothing = curr_smoothing)
        log_per = ngram_model.get_total_perplexity(log=calculate_log_perplexity)
        
        if isinstance(log_per, str):
            smoothing_outcomes.append(log_per)
        else:
            smoothing_outcomes.append(pow(10, log_per))

    comparsionTable.add_row(smoothing_outcomes)

print(comparsionTable)

with open('Smoothing_Comparison.txt', 'w') as outputFile:
    outputFile.write(str(comparsionTable))

'''
current_smoothing = None
print("Smoothing : ", current_smoothing)

#ngram models
unigram_model = LanguageModel(train_data, test_data, 1, smoothing = current_smoothing)
print("Unigram Log Perplexity: ", unigram_model.get_total_perplexity(log=calculate_log_perplexity))
print("Unigram Perplexity: ", pow(10, unigram_model.get_total_perplexity(log=calculate_log_perplexity)))

bigram_model = LanguageModel(train_data, test_data, 2, smoothing = current_smoothing)
print("Bigram Log Perplexity: ", bigram_model.get_total_perplexity(log=calculate_log_perplexity))
print("Bigram Perplexity: ", pow(10, bigram_model.get_total_perplexity(log=calculate_log_perplexity)))

trigram_model = LanguageModel(train_data, test_data, 3, smoothing = current_smoothing)
print("Trigram Log Perplexity: ", trigram_model.get_total_perplexity(log=calculate_log_perplexity))
print("Trigram Perplexity: ", pow(10, trigram_model.get_total_perplexity(log=calculate_log_perplexity)))

quadgram_model = LanguageModel(train_data, test_data, 4, smoothing = current_smoothing)
print("Quadgram Log Perplexity: ", quadgram_model.get_total_perplexity(log=calculate_log_perplexity))
print("Quadgram Perplexity: ", pow(10, quadgram_model.get_total_perplexity(log=calculate_log_perplexity)))

pentagram_model = LanguageModel(train_data, test_data, 5, smoothing = current_smoothing)
print("Pentagram Log Perplexity: ", pentagram_model.get_total_perplexity(log=calculate_log_perplexity))
print("Pentagram Perplexity: ", pow(10, pentagram_model.get_total_perplexity(log=calculate_log_perplexity)))

hexagram_model = LanguageModel(train_data, test_data, 6, smoothing = current_smoothing)
print("Hexagram Log Perplexity: ", hexagram_model.get_total_perplexity(log=calculate_log_perplexity))
print("Hexagram Perplexity: ", pow(10, hexagram_model.get_total_perplexity(log=calculate_log_perplexity)))

septagram_model = LanguageModel(train_data, test_data, 7, smoothing = current_smoothing)
print("Septagram Log Perplexity: ", septagram_model.get_total_perplexity(log=calculate_log_perplexity))
print("Septagram Perplexity: ", pow(10, septagram_model.get_total_perplexity(log=calculate_log_perplexity)))
'''