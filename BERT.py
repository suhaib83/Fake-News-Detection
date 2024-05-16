import time
import datetime
import pandas as pd
import numpy as np
import json
import random
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
## Torch Modules
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import (
    BertForSequenceClassification,
                          BertTokenizer,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                         AdamW)

import logging
logging.basicConfig(level = logging.ERROR)
import ssl
from numpyencoder import NumpyEncoder

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


if torch.cuda.is_available():     
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



def tokenize_dataset(df, num_of_way):
    df = df.sample(frac=1).reset_index(drop=True)    
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))     
    sentences = df['clean_title'].values
    labels = df['{}_way_label'.format(num_of_way)].values


    print(' Original: ', sentences[0])    
    print('Tokenized BERT: ', bert_tokenizer.tokenize(sentences[0]))    
    print('Token IDs BERT: ', bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sentences[0])))

    max_len_bert = 0

    for sent in sentences:    
        input_ids_bert = bert_tokenizer.encode(sent, add_special_tokens=True)       
        max_len_bert = max(max_len_bert, len(input_ids_bert))

    print('Max sentence length BERT: ', max_len_bert)    
    bert_input_ids = []
    bert_attention_masks = []
    sentence_ids = []
    counter = 0

    # For every sentence...
    for sent in sentences:

        bert_encoded_dict = bert_tokenizer.encode_plus(
                            sent,                     
                            add_special_tokens = True, 
                            max_length = 120,          
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',   
                       )

         
        bert_input_ids.append(bert_encoded_dict['input_ids'])    
        bert_attention_masks.append(bert_encoded_dict['attention_mask'])

        sentence_ids.append(counter)
        counter  = counter + 1

    bert_input_ids = torch.cat(bert_input_ids, dim=0)
    bert_attention_masks = torch.cat(bert_attention_masks, dim=0)      

    labels = torch.tensor(labels)
    sentence_ids = torch.tensor(sentence_ids)

    torch.manual_seed(0)
    bert_dataset = TensorDataset(sentence_ids, bert_input_ids, bert_attention_masks, labels)
    return bert_dataset



def index_remover(tensordata):
    input_ids = []
    attention_masks = []
    labels = []
   
    for a,b,c,d in tensordata:
        input_ids.append(b.tolist())
        attention_masks.append(c.tolist())
        labels.append(d.tolist())
        
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(labels)
    
    final_dataset =  TensorDataset(input_ids, attention_masks, labels)
    return final_dataset



def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Multimodal dataset
df_train = pd.read_csv('../Data/multimodal_train.csv')
df_val = pd.read_csv('../Data/multimodal_valid.csv')
df_test = pd.read_csv('../Data/multimodal_test.csv')

# clean NaN in clean titles
df_train = df_train[df_train['clean_title'].notna()]
df_val = df_val[df_val['clean_title'].notna()]
df_test = df_test[df_test['clean_title'].notna()]



num_of_way = 6 

# BERT
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                                num_labels = num_of_way,  
                                                                output_attentions = False, 
                                                                output_hidden_states = False 
                                                          )

bert_model.to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(' BERT model loaded')
bert_train_dataset = tokenize_dataset(df_train,num_of_way)
bert_val_dataset = tokenize_dataset(df_val,num_of_way)
bert_train_dataset = index_remover(bert_train_dataset)
bert_val_dataset = index_remover(bert_val_dataset)

batch_size = 32


bert_train_dataloader = DataLoader(
            bert_train_dataset,  
            sampler = RandomSampler(bert_train_dataset), 
            batch_size = batch_size 
        )

bert_validation_dataloader = DataLoader(
            bert_val_dataset,
            sampler = SequentialSampler(bert_val_dataset), 
            batch_size = batch_size 
        )

bert_optimizer = AdamW(bert_model.parameters(),
                  lr = 5e-5, 
                  eps = 1e-8 
                )

epochs = 10
skip_train = False

total_steps = len(bert_train_dataloader) * epochs

bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


seed_val = 100

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

bert_training_stats = []


total_t0 = time.time()


if not skip_train:
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        total_train_loss = 0
        bert_model.train()

        for step, batch in enumerate(bert_train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)

                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(bert_train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            bert_model.zero_grad()

            outputs = bert_model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits


            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            bert_optimizer.step()
            bert_scheduler.step()

        
        avg_train_loss = total_train_loss / len(bert_train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()

        bert_model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        batch_counter = 0
        for batch in bert_validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            with torch.no_grad():
                output= bert_model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                loss, logits = outputs.loss, outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(bert_validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(bert_validation_dataloader)

        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        bert_training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

if skip_train:  
    # bert_model = torch.load("bert_model_save")
    bert_model = BertForSequenceClassification.from_pretrained('bert_save_dir/')
else:  
    bert_model.save_pretrained('bert_save_dir/')
    with open('bert_training_stats.txt', 'w') as filehandle:
        json.dump(bert_training_stats, filehandle)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=bert_training_stats)
    df_stats = df_stats.set_index('epoch')
    print(df_stats)

print('Number of test sentences: {:,}\n'.format(df_test.shape[0]))
sentences = df_test['clean_title'].values
labels = df_test['{}_way_label'.format(num_of_way)].values

input_ids = []
attention_masks = []

for sent in sentences:

    encoded_dict = bert_tokenizer.encode_plus(
                        sent,                      
                        add_special_tokens = True, 
                        max_length = 75,          
                        pad_to_max_length = True,
                        return_attention_mask = True,   
                        return_tensors = 'pt',
                   )
     
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])


input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)
  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))


bert_model.eval()
predictions , true_labels = [], []
 
for batch in prediction_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = bert_model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)  

    logits = outputs[0]  
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    predictions.append(logits)
    true_labels.append(label_ids)

print('DONE.')
print('Positive samples: %d of %d (%.2f%%)' % (df_test['{}_way_label'.format(num_of_way)].sum(), len(df_test['{}_way_label'.format(num_of_way)]), (df_test['{}_way_label'.format(num_of_way)].sum() / len(df_test['{}_way_label'.format(num_of_way)]) * 100.0)))


flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
df_test['{}_way_pred'.format(num_of_way)] = flat_predictions
flat_true_labels = np.concatenate(true_labels, axis=0)

def get_eval_report(labels, preds):
    cm = confusion_matrix(labels, preds) 
    print('Confusion Matrix:') 
    print(cm)
    cr = classification_report(labels, preds)
    print('\nClassification Report:') 
    print(cr)


eval_report = get_eval_report(flat_true_labels, flat_predictions)







