import random
import pandas as pd
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import LabelSentenceReader, InputExample
from torch.utils.data import DataLoader
import csv
import gzip
import os
import argparse


parser = argparse.ArgumentParser("SBERT training")
parser.add_argument('--epochs', type = int, default = 10, \
    help = 'number of training for Sentence BERT model')

# Load pre-trained model - we are using the original Sentence-BERT for this example.
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


class LabelSentenceReader:
    """Reads in a file that has at least two columns: a label and a sentence.
    This reader can for example be used with the BatchHardTripletLoss.
    Maps labels automatically to integers"""
    def __init__(self, folder, label_col_idx=1, sentence_col_idx=0, separator=','):
        self.folder = folder
        self.label_map = {}
        self.label_col_idx = label_col_idx
        self.sentence_col_idx = sentence_col_idx
        self.separator = separator

    def get_examples(self, filename, max_examples=0, encoding="utf-8"):
        examples = []

        id = 0
        data = pd.read_csv(os.path.join(self.folder, filename), encoding = encoding)
        
        for i in range(len(data)):
            label = data.iloc[i,self.label_col_idx][-3:]
            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)
            label_id = self.label_map[label]
            sentence = data.iloc[i,self.sentence_col_idx]

            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence], label=label_id))

            if 0 < max_examples <= id:
                break

        return examples


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets


if __name__ =='__main__':
    args = parser.parse_args()

    # Set up data for fine-tuning 
    workdir = os.getcwd()

    sentence_reader = LabelSentenceReader(folder= workdir)
    data_list = sentence_reader.get_examples(filename = 'data/SBERT_train.csv',
                                            encoding = "ISO-8859-1")
    triplets = triplets_from_labeled_dataset(input_examples=data_list)
    finetune_data = SentencesDataset(examples=triplets, model=sbert_model)
    finetune_dataloader = DataLoader(finetune_data, shuffle=True, batch_size=16)

    # Initialize triplet loss
    loss = TripletLoss(model=sbert_model)

    # Fine-tune the model
    sbert_model.fit(train_objectives=[(finetune_dataloader, loss)], \
        epochs=args.epochs, 
        save_best_model= True,
        output_path= os.path.join(workdir, 'SBERT/bot'))