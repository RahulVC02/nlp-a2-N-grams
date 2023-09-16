import math
from utils import preprocess, make_ngrams_dict, get_ngrams_list_from_sentence, freq_calc, goodTuring_ngram_count


class LanguageModel(object):
    def __init__(self, train_data, test_data, n, smoothing=None):
        self.n = n
        self.smoothing = smoothing
        self.preprocessed_sentences = preprocess(train_data, n)
        self.preprocessed_test_sentences = preprocess(test_data, n)
        self.vocab = make_ngrams_dict(self.preprocessed_sentences, 1)
        self.num_words = sum(self.vocab.values())
        self.vocab_size = len(self.vocab)
        self.ngrams_dict = make_ngrams_dict(self.preprocessed_sentences, n)
        self.freq_dict = freq_calc(self.ngrams_dict)
        if(n==1):
            self.n_minus_one_grams_dict = {}
        else:
            self.n_minus_one_grams_dict = make_ngrams_dict(self.preprocessed_sentences, n-1)
    
    def _predict_ngram_prob(self, ngram):
        ngram_string = " ".join(ngram)
        if(self.smoothing is None or self.smoothing != 'goodTuring'):
            ngram_count = self.ngrams_dict.get(ngram_string, 0)
            if(self.n>1):
                n_minus_one_gram = ngram[:-1]
                n_minus_one_gram_string = " ".join(n_minus_one_gram)
                n_minus_one_gram_count = self.n_minus_one_grams_dict.get(n_minus_one_gram_string, 0)
            else:
                n_minus_one_gram_count = self.num_words
            
            if(self.smoothing is None):
                try:
                    prob = ngram_count / n_minus_one_gram_count
                    return prob
                except ZeroDivisionError:
                    return None
            elif(self.smoothing == 'laplace'):
                return (ngram_count + 1) / (n_minus_one_gram_count + self.vocab_size)
            else:                                                                       #ADD-K Smoothing
                return (ngram_count + self.smoothing) / (n_minus_one_gram_count + self.vocab_size * self.smoothing)
        else:
            goodTuringNgramCount = goodTuring_ngram_count(ngram_string, self.ngrams_dict, self.freq_dict)
            num_ngrams_in_training_corpus = sum(self.ngrams_dict.values())
            return goodTuringNgramCount / num_ngrams_in_training_corpus         
        
    def _get_sentence_perplexity(self, sentence, log):
        N = len(sentence.split())
        ngrams_list = get_ngrams_list_from_sentence(sentence, self.n)
        if(log):
            sentence_log_probability = 0
            constc = -1/N
            for ngram in ngrams_list:
                prob = self._predict_ngram_prob(ngram)
                if(prob == 0 or prob is None):
                    return None
                sentence_log_probability += math.log10(prob)
            return constc * sentence_log_probability
        else:
            sentence_perplexity = 1
            for ngram in ngrams_list:
                sentence_perplexity /= self._predict_ngram_prob(ngram)
            return math.pow(sentence_perplexity, 1/N)
    
    def get_total_perplexity(self, log=False):
        test_sentences = self.preprocessed_test_sentences
        num_sentences = len(test_sentences)
        total_perplexity = 0
        for sentence in test_sentences:
            sentence_perplexity = self._get_sentence_perplexity(sentence, log)
            if(sentence_perplexity is None):
                return "Infinite"
            total_perplexity += sentence_perplexity
        return total_perplexity / num_sentences