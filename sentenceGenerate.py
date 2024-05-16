import numpy as np
import tensorflow.compat.v1 as tf
import random
from Discriminator import Discriminator
from LeakGANModel import  LeakGAN
import pickle as cPickle
from convertor import convertor
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

flags = tf.app.flags
FLAGS = flags.FLAGS

EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 32  # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 80  # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 48
GOAL_SIZE = 16
STEP_SIZE = 4

#  Discriminator  Hyper-parameters
dis_embedding_dim = 64

dis_filter_sizes = [2,3]
dis_num_filters = [100,200]
GOAL_OUT_SIZE = sum(dis_num_filters)

dis_dropout_keep_prob = 1.0
dis_l2_reg_lambda = 0.2

#  Basic Training Parameters
TOTAL_BATCH = 200
generate_file = 'save_generator/result_sentenceGenerate.txt'
pickle_loc = 'data/vocab_py2.pkl' 
generated_num = 5
model_path = './ckpts'
maxModelSave = 30


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, train = 1):
    # Generate Samples
    tf.disable_eager_execution()
    generated_samples = []
    round = int(generated_num / batch_size) + 1

    for i in range(round):
        generated_samples.extend(trainable_model.generate(sess,1.0,train))

    with open(output_file, 'w') as fout:
        count = 0
        for Fakenews in generated_samples:
            buffer = ' '.join([str(x) for x in Fakenews]) + '\n'
            fout.write(buffer)
            count += 1
            if count == generated_num:
                break
            
def generate_samples_updated(sess, trainable_model, batch_size, generated_num, train=1):
    # Generate Samples
    tf.disable_eager_execution()
    generated_samples = []
    round = int(generated_num / batch_size) + 1

    for i in range(round):
        generated_samples.extend(trainable_model.generate(sess, 1.0, train))

    # Convert generated samples to strings
    generated_strings = []
    count = 0
    for Fakenews in generated_samples:
        buffer = ' '.join([str(x) for x in Fakenews])
        generated_strings.append(buffer)
        count += 1
        if count == generated_num:
            break

    return generated_strings

def coverNieun(line):
    newSentence = []
    line = line.split(' ')

    for i in line:

        # ㄴ Merger
        merge = False

        if i == "ㄴ":

            # If there is a word in front and the last letter of the word is Hangul,
            if len(newSentence) != 0 and ord(newSentence[-1][-1]) >= 44032 and ord(newSentence[-1][-1]) < 55200:

                # If the last letter of the preceding word has no ending,
                if (ord(newSentence[-1][-1]) - 44032) % 28 == 0:
                    # print(chr(ord(newSentence[-1][-1])+4))
                    newSentence[-1] = newSentence[-1][:-1] + chr(ord(newSentence[-1][-1]) + 4)
                    merge = True

                elif (ord(newSentence[-1][-1]) - 44032) % 28 == 17:
                    newSentence[-1] = newSentence[-1][:-1] + chr(ord(newSentence[-1][-1]) - 17) + ','
                    merge = True

        if not merge:
            newSentence.append(i)

    return ' '.join(newSentence)

def generation_txt(generated_num):
    tf.disable_eager_execution()
    random.seed(SEED)
    np.random.seed(SEED)
    with open(pickle_loc, 'rb') as f: 
        _, _, _, SEQ_LENGTH, vocab_size = cPickle.load(f)

    
    assert START_TOKEN == 0
    discriminator = Discriminator(SEQ_LENGTH,num_classes=2,vocab_size=vocab_size,dis_emb_dim=dis_embedding_dim,filter_sizes=dis_filter_sizes,num_filters=dis_num_filters,
                        batch_size=BATCH_SIZE,hidden_dim=HIDDEN_DIM,start_token=START_TOKEN,goal_out_size=GOAL_OUT_SIZE,step_size=4)
    
    leakgan = LeakGAN(SEQ_LENGTH,num_classes=2,vocab_size=vocab_size,emb_dim=EMB_DIM,dis_emb_dim=dis_embedding_dim,filter_sizes=dis_filter_sizes,num_filters=dis_num_filters,
                        batch_size=BATCH_SIZE,hidden_dim=HIDDEN_DIM,start_token=START_TOKEN,goal_out_size=GOAL_OUT_SIZE,goal_size=GOAL_SIZE,step_size=4,D_model=discriminator)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver_variables = tf.global_variables()
    saver = tf.train.Saver(saver_variables, max_to_keep=maxModelSave)
    print ("start sentence generate!!")
    generate_samples(sess, leakgan, BATCH_SIZE, generated_num, generate_file, 0)
    print ("convertor sentence generate!!")
    convertor(generate_file, filedir='save_generator/')
    print ("sentenceGenerate.py finish!")
