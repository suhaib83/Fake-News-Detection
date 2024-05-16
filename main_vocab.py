import random
import codecs
import pickle
import tensorflow as tf


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eof_code = len(dictionary)
    for sentence in tokens:
        index = 0
        sentence = sentence.strip().split(" ")
        for word in sentence:
            code_str += str(dictionary[word]) + ' '
            index += 1
        while index < seq_len:
            code_str += (str(eof_code) + ' ')
            index += 1
        code_str += '\n'
    return code_str


def code_to_text(codes, dictionary):
    paras = ""
    eof_code = len(dictionary)
    for line in codes:
        line = str(line).strip().split(' ')
        numbers = map(int, line)
        for number in numbers:
            if number == eof_code:
                continue
            paras += (dictionary[str(number)] + ' ')
        paras += '\n'
    return paras


def get_tokenlized(file, num_readline):
    tokenlized = list()

    raw = codecs.open(file, 'r', encoding='utf-8')
    if num_readline != "ALL":
        num_read = random.sample(range(1, len(raw.readlines()) - 1), num_readline)
    raw.closed

    selectedsentence = list()
    count = 0
    with open(file, 'r', encoding='utf-8') as raw:
        raw.readline()
        for text in raw:
            if num_readline == "ALL":
                selectedsentence.append(text)
                tokens = text.strip().split(" ")
                tokenlized.append(tokens)
            else:
                if count in num_read:
                    selectedsentence.append(text)
                    tokens = text.strip().split(" ")
                    tokenlized.append(tokens)
                    # text = nltk.word_tokenize(text.lower())
                    # tokenlized.append(text)
                count += 1

    f = open('data/selected.txt', 'w', encoding='utf-8')
    for i in selectedsentence:
        f.write(str(i))
    f.closed
    return tokenlized

def get_word_list(tokens):
    word_set = list()
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()
    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict

def text_precess(train_text_loc, num_readline, test_text_loc=None):
    train_tokens = get_tokenlized(train_text_loc, num_readline)

    if test_text_loc is None:
        test_tokens = list()
    else:
        test_tokens = get_tokenlized(test_text_loc, num_readline)
    word_set = get_word_list(train_tokens + test_tokens)

    [word_index_dict, index_word_dict] = get_dict(word_set)

    if test_text_loc is None:
        sequence_len = len(max(train_tokens, key=len))
    else:
        sequence_len = max(len(max(train_tokens, key=len)), len(max(test_tokens, key=len)))
        with open('data/eval_data.txt', 'w', encoding='utf-8') as outfile:
            outfile.write(text_to_code(test_tokens, word_index_dict, sequence_len))

    with open('data/vocab_py2.pkl', 'wb') as f:
        pickle.dump([index_word_dict, word_index_dict, word_set, sequence_len, len(word_index_dict) + 1], f, protocol=2)

    with open('data/vocab_py3.pkl', 'wb') as f:
        pickle.dump([index_word_dict, word_index_dict, word_set, sequence_len, len(word_index_dict) + 1], f, protocol=3)

    return sequence_len, len(word_index_dict) + 1


data_loc = 'data/FakeNews.txt'

def build_vocab():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    # num_readline = (int) or "ALL"
    num_readline = "ALL"
    SEQ_LENGTH, vocab_size = text_precess(data_loc, num_readline)
    print('SEQ_LENGTH :', SEQ_LENGTH)
    print('vocab_size :', vocab_size)
    
    index_word_dict, word_index_dict, word_set, SEQ_LENGTH, vocab_size = pickle.load(open('data/vocab_py3.pkl', 'rb'))
    
    with open('data/text_to_code.txt', 'w', encoding='utf-8') as outfile:
        with open('data/selected.txt', 'r', encoding='utf-8') as readfile:
            outfile.write(text_to_code(readfile.readlines(), word_index_dict, SEQ_LENGTH))
