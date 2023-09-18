SOS = "<s> "
EOS = "</s>"

def add_sentence_tokens(sentences, n):
    sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

def preprocess(sentences, n):
    processed_sentences = add_sentence_tokens(sentences, n)
    return processed_sentences

def make_ngrams_dict(processed_sentences, n):
    counts = {}
    for sentence in processed_sentences:
        counts = counter(sentence, n, counts)
    return counts

def get_ngrams_list_from_sentence(sentence, n):
    ngrams_list = []
    tokens = sentence.split()
    for i in range(len(tokens)-n+1):
        ngrams_list.append(tokens[i:i+n])
    return ngrams_list

def counter(data, n, count_gram_dict):        
    vocab = data.split()
    gram_gram = [""]
    gram_gram.extend(vocab[:n-1])
    for i in range (n, len(vocab)+1):
        gram_gram.pop(0)
        gram_gram.append(vocab[i-1])
        gram_element = " ".join(gram_gram)
        count_gram_dict[gram_element] = count_gram_dict.get(gram_element,0) + 1
    return count_gram_dict

def freq_calc(count_gram_dict):
    freq_dict = {}
    for key in count_gram_dict:
        freq_dict[count_gram_dict[key]] = freq_dict.get(count_gram_dict[key],0) + 1
    return freq_dict

def goodTuring_ngram_count(ngram, count_gram_dict, freq_dict, tot_grams):
    c = count_gram_dict.get(ngram,0)
    if c > 100 or freq_dict.get(c+1,0)==0:    #done for maintaining consistency after the count becomes large, can be done for different values for different models
                                              #although for our set freq[c+1] was never 0 but put for safety
        return c - 0.75
    elif (c == 0):
        return freq_dict[1]/min((tot_grams - len(count_gram_dict)),1e6)
    else:
        return (c+1)*freq_dict.get(c+1,0)/freq_dict[c]

def tot_vocab(vocab, sentences):
    words = make_ngrams_dict(sentences, 1)
    count = len(vocab)
    for word in words:
        if word not in vocab:
            count += 1
    return count