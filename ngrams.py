import math
from utils import preprocess, make_ngrams_dict, get_ngrams_list_from_sentence

class LanguageModel(object):
    def __init__(self, train_data, test_data, n, smoothing='laplace'):
        self.n = n
        self.smoothing = smoothing
        self.preprocessed_sentences = preprocess(train_data, n)
        self.preprocessed_test_sentences = preprocess(test_data, n)
        self.vocab = make_ngrams_dict(self.preprocessed_sentences, 1)
        self.vocab_size = len(self.vocab)
        self.ngrams_dict = make_ngrams_dict(self.preprocessed_sentences, n)
        if(n==1):
            self.n_minus_one_grams_dict = {}
        else:
            self.n_minus_one_grams_dict = make_ngrams_dict(self.preprocessed_sentences, n-1)
    
    def _predict_ngram_prob(self, ngram):
        ngram_string = " ".join(ngram)
        ngram_count = self.ngrams_dict.get(ngram_string, 1)
        if(self.n>1):
            n_minus_one_gram = ngram[:-1]
            n_minus_one_gram_string = " ".join(n_minus_one_gram)
            n_minus_one_gram_count = self.n_minus_one_grams_dict.get(n_minus_one_gram_string, self.vocab_size)
        else:
            n_minus_one_gram_count = self.vocab_size
        return ngram_count / n_minus_one_gram_count
        
    def _get_sentence_perplexity(self, sentence, log):
        N = len(sentence.split())
        ngrams_list = get_ngrams_list_from_sentence(sentence, self.n)
        if(log):
            sentence_perplexity = 0
            constc = -1/N
            for ngram in ngrams_list:
                sentence_perplexity += math.log(self._predict_ngram_prob(ngram))
            return constc * sentence_perplexity
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
            total_perplexity += self._get_sentence_perplexity(sentence, log)
        return total_perplexity / num_sentences