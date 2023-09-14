import nltk
SOS = "<s> "
EOS = "</s>"
UNK = "<UNK>"

def add_sentence_tokens(sentences, n):
    sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

def replace_singletons(tokens):
    #TODO Replace with custom FreqDist implementation
    vocab = nltk.FreqDist(tokens)       
    return [token if vocab[token] > 1 else UNK for token in tokens]

def preprocess(sentences, n):
    sentences = add_sentence_tokens(sentences, n)
    tokens = ' '.join(sentences).split(' ')
    tokens = replace_singletons(tokens)
    return tokens


def load_data(data_dir):
    train_path = data_dir + 'train.txt'
    test_path  = data_dir + 'test.txt'

    with open(train_path, 'r') as f:
        train = [l.strip() for l in f.readlines()]
    with open(test_path, 'r') as f:
        test = [l.strip() for l in f.readlines()]
    return train, test


#Add Frequency Distribution code to
#the codebase