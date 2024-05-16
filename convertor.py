import pickle

def convertor(file, filedir):

    vocab_file = "data/vocab_py2.pkl"
    output_file_tag = 'text'

    word, vocab, _, _, _ = pickle.load(open(vocab_file, 'rb'))
    pad_num = len(word)

    input_file = file
    output_file = filedir + output_file_tag + '_' + file.split('_')[-1]
    with open(output_file, 'w', encoding='utf-8')as fout:
        with open(input_file, 'r')as fin:
            for line in fin:
                line = line.split()
                words = list()
                for x in line:
                    if x == str(pad_num):
                        break
                    else:
                        data = word[x]
                        words.append(data)
                line = words
                line = ' '.join(line) + '\n'
                fout.write(line)

    output_file_notag = filedir + output_file_tag + '_notag_' + file.split('_')[-1]
    with open(output_file_notag, 'w',encoding='utf-8')as fout:
        with open(input_file, 'r')as fin:
            for line in fin:
                line = line.split()
                words = list()
                for x in line:
                    if x == str(pad_num):
                        break
                    else:
                        data = word[x]
                        data = data[:data.rfind('/')]
                        words.append(data)
                line = words
                line = ' '.join(line) + '\n'
                fout.write(line)
