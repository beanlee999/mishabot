# Introduction

The name of the chatbot is *Misha*, which stands for "My Information Service and Health Assistant". She is a  female conversational bot in her thirties, designed to provide the Covid-19 guidelines for residents living in Singapore in 2021, when Singapore was navigating through a period of rapid changes in Covid guidelines. In addition, she is able to provide short responses to chats initiated by users as well.

The objectives of Misha are as follows:
-	Provide quick and direct answers to user queries on Covid guidelines in Singapore relating to health, immigration and schools
-	Enhance user experience by providing greater accuracies in Question-Answer for Covid-19 related guidelines in Singapore

Misha is currently available for deployment on local devices (laptops and computers). She is trained mainly from FAQ datasets extracted from Ministry of Health, Ministry of Education and Immigration and Customs Authority websites. While the main purpose of Misha is to provide answers to user queries, she can respond briefly to users’ small chat as well. 

# Design
The flow of the Misha chatbot is as such:

1.	User enters a text.
2.	Misha will interpret if user’s intent is to end conversation. If yes, Misha will end conversation, else, it will move to next step.
3.	Misha will identify if entered text contains keywords related to the FAQ datasets. If no, it will interpret text as a casual remark and will respond with small talk conversation by computing cosine similarity score after encoding using SentenceTransformers.
4.	If text contains keywords, Misha will check if the text is a question. If the text is not a question, Misha will ask if user would like to pose a question instead (Figure 28 in the appendix).
5.	If text is a question, Misha will compute cosine similarity between encoded text and each of the FAQ question, find the closest question match and seek feedback from user if it is what he or she is looking for. If yes, Misha will return the corresponding answer, else, Misha will ask user to input question again.

<img src="https://github.com/beanlee999/mishabot/blob/cy/assets/design.png" alt="BERT classifier model" width="778" height="800"/>

# Repository Structure

The structure of the repository is as below:

```
project_repo
    ├── assets/             <- Images used for README.md
    ├── SBERT/              <-  Folder containing codes for sentence transformers
    ├── weights/            <-  Folder containing weights for classification model
    |                       (`classification.pt`)
    ├── codes/              <-  Folders containing codes for training 
    │   ├── classification.py
    |   |                 ^- Codes to load `weights/classification.pt`, takes in embedded user
    |   |                   input and return results as a question or not
    |   ├── classification_train.py
    |   |                 ^- Codes to train BERT classification model (question or not)
    |   ├── faq_functions.py
    |   |                 ^- Codes containing conversational, Q&A element of Misha
    |   ├── faq_functions_tts.py
    |   |                 ^- Codes containing conversational, Q&A element of Misha
    |   |                 with text-to-speech element
    │   └── faq_SBERT_train.py
    |                     ^- codes to train Sentence Transformers model and save trained results in 
    |                       `SBERT/bot` folder
    ├── data/               <-  Folders containing text datasets for training sentence transformers and
    |                       BERT classification model
    ├── README.md           <-  Explains working principle of the Chatbot
    │                       and how to set it up
    ├── requirements.yml    <-  YAML file containing dependency list for setting up
    │                       conda environment
    ├── chatbot.py          <-  Python scripts to set up Misha Chatbot
    ├── chatbot_tts.py      <-  Python scripts to run Misha Chatbot with voice annotation
    └── .gitignore          <-  File for specifying files or directories
                            to be ignored by Git.

```

# Steps to set up
To set up and run the chatbot:

1. In windows, open anaconda prompt. In Mac, open terminal. 
2. Change working directory to chatbot project folder directory ("cd <filepathtofolder>/chatbot_v4")
3. Create a conda environment using `requirements.yml` [`conda env create -f requirements.yml`]. The name of the conda environment is `mishabot`
4. Activate conda env (`conda activate mishabot`)
5. ***Download classification model weights/train Sentence transformers model***
6. If you would like to run bot without any voice annotation, type (`python -m chatbot`). To run bot with voice annotation, type (`python -m chatbot_tts`)

# Data Preparation

There are a total of 6 datasets used to train Misha. 

For the Question-Answer feature, FAQs were extracted from Ministry of Health, Immigration and Customs Authority and Ministry of Education websites. The number of question-answer pairs in each dataset were 50, 52 and 139 respectively. As the dataset sizes were small, each question was paraphrased to generate more questions of the same type. A list of common keywords were extracted for each question too. A total of 216 unique keywords were identified, which were then used to identify if a chat input is related to Covid-19.

To train the classifer model to determine if a user input is a question, the SMS dataset and the Stanford Question and Answer Dataset (SQuAD) dataset were used. The SMS dataset contained SMSes from Singapore. During pre-training, SMS text containing "?" were classified as 1 (question) else 0 (not a question). For SQuAD, the dataset used comes with labels of 1 (question) and 0 (non-question).

A small dataset referenced from Google Dialogflow(TM)'s smalltalk agent was created for the purpose of incorporating chat element into Misha to provide a sense of personal touch and engagement to users so that Misha feels more human-like.

<img src="https://github.com/beanlee999/mishabot/blob/cy/assets/datasets.png" alt="Sentence Transformers" width="756" height="400"/>

# Working Principle
## Document Similarity Method

When a user enters an input, Misha needs to generate a response. This response is derived from either the FAQ datasets comprising of MOH, ICA and MOE FAQs datasets, or the smalltalk dataset (depending if user input is Covid related)

We finetuned SentenceTransformers, and used it to embed the 'question' column of the `FAQ.csv` and `small_talk.csv`, as well as user's chat input. SentenceTransformers is a form of BERT sentence embedding, which has been used to compute sentence or text embeddings for more than 100 languages. We finetuned this model from the pre-trained model (*all-MiniLM-L6-v2*). This finetuning step was necessary as the pretrained model's training dataset does not include Covid related terms e.g. Covid-19, MOE, MOH, antigen rapid test etc). 

The fine-tuned model was used to encode each question in the FAQ datasets and the user’s input. The resulting embedding is a sentence level/text level embedding. Cosine similarity was computed between each encoded question and the encoded user’s input to find the question with the highest cosine similarity score (Figure 19). A user check was performed to ensure that the closest question match was what he/she was looking for. If it is a match, Misha will return the corresponding answer to the matched question.

<img src="https://github.com/beanlee999/mishabot/blob/cy/assets/sentencetransformers.png" alt="Sentence Transformers" width="505" height="518"/>

## Question Classifier

When a user inputs a text to Misha, he could pass a casual remark about Covid-19 which does not necessitate an answer from Misha. Therefore, a layer of checking is needed to identify the user’s intent (question or remark) in order for Misha to provide the right kind of response.

We used a classifier model that is built on basis of BERT-base model and further fine-tuned. The tokeniser used is BERT tokeniser (bert-base-uncased). It is used to tokenise the user’s input to generate token_ids, segment embedding and positional embedding, which were both passed to the BERT model. The model architecture is the same as that used for BERT sentiment classifier (Figure 8). The predicted outcome with the highest probability is the selected result (1 being a question, 0 being non-question).

<img src="https://github.com/beanlee999/mishabot/blob/cy/assets/BERTmodel.png" alt="BERT model" width="336" height="500"/>

Model weights are fine-tuned using the SQuAD train dataset and evaluated on 10% of dataset comprising SMS and FAQ dataset. The evaluation test set’s accuracy is 65% and the f1-score for question and non-question is at 0.60 and 0.68 respectively. A second fine tuning was done with a training dataset comprising SQuAD train, SMS and FAQ dataset. The evaluation test set is 10% of SQuAD validation dataset. The accuracy on the test set is 100% and f1-score for question and non-question are both 1.00. The saved model weights are loaded into Misha to classify user intent into question and non-question. A total of 5 epochs was used to finetune the model weights.

<img src="https://github.com/beanlee999/mishabot/blob/cy/assets/classifier_results.png" alt="BERT classifier model" width="546" height="400"/>


# Model Training

These datasets are used to train 2 models:
- BERT classification model to determine if user input is a question or not
- Sentence transformers model to embed user input into a sentence embedding. Dataset used to trained this model is `data/SBERT_train.csv`, which contains a list of questions, question categories and the response to return. 

## Sentence Transformers Model

To train Sentence transformers model, use the following command:
```
python -m codes.faq_SBERT_train --epochs=10
```

## Classification Model 
To train classifier model, use the following command:
```
python -m codes.classification_train --epochs=10 --maxlen=35 --batch_size=128
```
**Note**: This could take a few hours to train.

# Demo
A version of Misha without text-to-speech is available as chatbot.py and a version with text-to-speech is available as chatbot_tts.py. Instructions on how to set up and run the Misha can be found in the README markdown file in the chatbot_v4 submission folder. A virtual environment is required to be set up and the necessary dependencies installed.

To run Misha without text-to-speech annotation:
```
python -m chatbot
```

To run Misha with text-to-speech annotation:
```
python -m chatbot_tts
```

<iframe
    width="640"
    height="480"
    src="https://youtu.be/wFFYXVhNHCg"
    frameborder="0"
    allow="autoplay; encrypted-media"
    allowfullscreen
>
</iframe>

<a href="https://youtu.be/wFFYXVhNHCg" target="_blank">
 <img src="https://i9.ytimg.com/vi/wFFYXVhNHCg/mq1.jpg?sqp=CIze2ZYG&rs=AOn4CLBQpoYTKYRANJiNEkfD4bt1kGdEOQ" alt="Misha Demo" width="640" height="480" border="10" />
</a>

[![Misha](https://i9.ytimg.com/vi/wFFYXVhNHCg/mq1.jpg?sqp=CIze2ZYG&rs=AOn4CLBQpoYTKYRANJiNEkfD4bt1kGdEOQ)](https://youtu.be/wFFYXVhNHCg "Misha")

# Credits
Credits to the following teammates who worked as a team (collaborators) to complete this chatbot:
- Lim Weijie
- Yap Wan Yi
- Foo Xiang Yu

And to Ms Tan Ming Shan for giving Misha a name (My Information Service and Health Assistant)