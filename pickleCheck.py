import pickle
index_word_dict, word_index_dict, word_set, SEQ_LENGTH, vocab_size = pickle.load(open('data/vocab_py3.pkl', 'rb'))
print(index_word_dict)
print(word_index_dict)
print(word_set)
print(SEQ_LENGTH)
print(vocab_size)