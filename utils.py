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
