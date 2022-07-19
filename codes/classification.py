import numpy as np
import torch
import os
import torch.nn as nn
import transformers
from transformers import AutoModel, BertTokenizerFast
import transformers
transformers.logging.set_verbosity_error()

# Import BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False
    
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
  
# pass the pre-trained BERT to our define architecture
model = BERT_Model(bert)

# push the model to GPU
model = model.to(device)


# Run Classification

class Classifier:
    def __init__(self, workdir):
        self.maxlen = 35
        self.workdir = workdir
        self.path = os.path.join(self.workdir, 'weights/classification.pt')

    def classify(self, userinput):
        tokens_train = tokenizer.batch_encode_plus(
            [userinput],
            max_length = self.maxlen,
            padding=True,
            truncation=True
        )
        
        input_seq = torch.tensor(tokens_train['input_ids'])
        input_mask = torch.tensor(tokens_train['attention_mask'])
        
        model.load_state_dict(torch.load(self.path, map_location = device))
        
        with torch.no_grad():
          preds = model(input_seq.to(device), input_mask.to(device))
          preds = preds.detach().cpu().numpy()
          
        preds = np.argmax(preds, axis = 1)
        return preds
