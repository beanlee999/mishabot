"""
This fine tuned model is adapted from the codes of:
https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/

Train dataset source: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
Test dataset source: 
    - SMS: https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/
    - FAQs: MOH, MOE, ICA FAQ websites
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import random
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizerFast, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

parser = argparse.ArgumentParser('BERT classification model training')
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--maxlen', type = int, default = 35)
parser.add_argument('--batch_size', type = int, default = 512)


# Define model architecture - to be customized!
class BERT_Model(nn.Module):
    def __init__(self, bert):
        super(BERT_Model, self).__init__()
        self.bert = bert 
        self.dropout = nn.Dropout(0.1)
        self.relu =  nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,2)
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      x = self.fc1(cls_hs) # first dense layer
      x = self.relu(x)      # Relu activation
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)

      return x

def train():
  """Method to train classification model

  Returns:
      _type_: _description_
  """
  model.train()
  total_loss = 0
  
  total_preds=[]
  
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    if step % 10 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
    batch = [r.to(device) for r in batch]
    sent_id, mask, labels = batch

    model.zero_grad()        
    preds = model.forward(sent_id, mask)
    loss = cross_entropy(preds, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    preds=preds.detach().cpu().numpy()
    total_preds.append(preds)

  avg_loss = total_loss / len(train_dataloader)
  total_preds  = np.concatenate(total_preds, axis=0)
  return avg_loss, total_preds


# Evaluate model
def evaluate():
  
  print("\nEvaluating...")
  model.eval()
  total_loss = 0
  
  total_preds = []
  for step,batch in enumerate(val_dataloader):
    if step % 10 == 0 and not step == 0:    
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
    batch = [t.to(device) for t in batch]
    sent_id, mask, labels = batch

    with torch.no_grad():
      preds = model.forward(sent_id, mask)
      loss = cross_entropy(preds,labels)
      total_loss = total_loss + loss.item()
      preds = preds.detach().cpu().numpy()
      total_preds.append(preds)

  avg_loss = total_loss / len(val_dataloader) 

  total_preds  = np.concatenate(total_preds, axis=0)
  return avg_loss, total_preds

   

if __name__ == '__main__':
  args = parser.parse_args()    
  
  # specify GPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Loading train data
  # data set is SQuAD train dataset (https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
  # Label: Question = 1, Answers = 0

  workdir = os.getcwd()
  SQuAD_train = pd.read_csv(os.path.join(workdir, 'data/SQuAD_train.csv'),
                            encoding = "ISO-8859-1")

  SQuAD_train[['Labels']] = SQuAD_train[['Labels']].astype(int)

  # data downloaded from https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/
  sms = pd.read_csv(os.path.join(workdir, 'data/SMS.txt')).drop(columns = 'label')
  faq = pd.read_csv(os.path.join(workdir, 'data/FAQs.csv'))

  # Creating Labels for SMS data - 1 for question, 0 for non-questions
  sms['Labels'] = [0 if x.find("?") == -1 else 1 for x in sms['text']]

  # Creating Labels for FAQs - 1 for question, 0 for non-questions
  faq['Labels'] = [0 if x.find("?") == -1 else 1 for x in faq.Question]

  sms = sms[['text', 'Labels']]
  faq = faq[['Question', 'Labels']]
  faq.columns = ['text', 'Labels']

  train_data = SQuAD_train.append(sms).append(faq).dropna()
  train_data = train_data.sample(frac = 1.0)
  train_data[["Labels"]] = train_data[["Labels"]].astype('int')


  test_data = pd.read_csv(os.path.join(workdir, 'data/SQuAD_test.csv'),
                          encoding = "ISO-8859-1")
  test_data = test_data.sample(frac = 1.0).dropna()
  test_data[["Labels"]] = test_data[["Labels"]].astype('int')


  # Split into train test validation sets
  # Split into train test validation sets
  x_train, x_val, y_train, y_val = train_test_split(train_data['text'], \
    train_data['Labels'], random_state =42, train_size=0.8, 
    stratify= train_data['Labels'])

  x_test, y_test = test_data['text'].values, test_data['Labels'].values

  # Import BERT model and tokenizer
  bert = AutoModel.from_pretrained('bert-base-uncased')

  # Load the BERT tokenizer
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

  # Tokenize and encode text
  tokens_train = tokenizer.batch_encode_plus(x_train.tolist(), \
    max_length = args.maxlen, padding=True, truncation=True)

  # tokenize and encode sequences in the validation set
  tokens_val = tokenizer.batch_encode_plus(x_val.tolist(), \
    max_length = args.maxlen, padding=True, truncation=True)

  # tokenize and encode sequences in the test set
  tokens_test = tokenizer.batch_encode_plus(x_test.tolist(),
      max_length = args.maxlen, padding=True, truncation=True)

  ## convert lists to tensors
  train_seq = torch.tensor(tokens_train['input_ids'])
  train_mask = torch.tensor(tokens_train['attention_mask'])
  train_y = torch.tensor(y_train.tolist())

  val_seq = torch.tensor(tokens_val['input_ids'])
  val_mask = torch.tensor(tokens_val['attention_mask'])
  val_y = torch.tensor(y_val.tolist())

  test_seq = torch.tensor(tokens_test['input_ids'])
  test_mask = torch.tensor(tokens_test['attention_mask'])
  test_y = torch.tensor(y_test.tolist())

  #define a batch size
  batch_size = args.batch_size

  # wrap tensors
  train_data = TensorDataset(train_seq, train_mask, train_y)
  # sampler for sampling the data during training
  train_sampler = RandomSampler(train_data)
  # dataLoader for train set
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  # wrap tensors
  val_data = TensorDataset(val_seq, val_mask, val_y)
  # sampler for sampling the data during validation
  val_sampler = SequentialSampler(val_data)
  # dataLoader for validation set
  val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

  # freeze all the parameters
  for param in bert.parameters():
      param.requires_grad = True
    
  # pass the pre-trained BERT to our define architecture
  model = BERT_Model(bert)

  # push the model to GPU
  model = model.to(device)

  # Define optimizers
  optimizer = AdamW(model.parameters(), lr = 1e-5)

  #compute the class weights
  class_weights = compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)

  print("Class Weights:",class_weights)

  # converting list of class weights to a tensor
  weights= torch.tensor(class_weights,dtype=torch.float)

  # push to GPU
  weights = weights.to(device)

  # define the loss function
  cross_entropy  = nn.NLLLoss(weight=weights) 

  # Train model
  epochs = args.epochs
  # set initial loss to 1000
  best_valid_loss = 1000.0

  # empty lists to store training and validation loss of each epoch
  train_losses=[]
  valid_losses=[]

  #for each epoch
  for epoch in range(epochs):
      
      print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
      #train model
      train_loss, _ = train()
      #evaluate model
      valid_loss, _ = evaluate()
      
      #save the best model
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(model.state_dict(), 
                    os.path.join(workdir, 'weights/classification.pt'))
      
      # append training and validation loss
      train_losses.append(train_loss)
      valid_losses.append(valid_loss)
      
      print(f'\nTraining Loss: {train_loss:.3f}')
      print(f'Validation Loss: {valid_loss:.3f}')

  # Performs evaluation on test set
  path = os.path.join(workdir, 'weights/classification.pt')
  model.load_state_dict(torch.load(path))

  with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()
    
  preds = np.argmax(preds, axis = 1)
  print(classification_report(test_y, preds))