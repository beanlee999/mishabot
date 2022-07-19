# Introduction

The name of the chatbot is *Misha*, which stands for "My Information Service and Health Assistant". She is a  female conversational bot in her thirties, designed to provide the Covid-19 guidelines for residents living in Singapore in 2021, when Singapore was navigating through a period of rapid changes in Covid guidelines. In addition, she is able to provide short responses to chats initiated by users as well.

The objectives of Misha are as follows:
-	Provide quick and direct answers to user queries on Covid guidelines in Singapore relating to health, immigration and schools
-	Enhance user experience by providing greater accuracies in Question-Answer for Covid-19 related guidelines in Singapore

Misha is currently available for deployment on local devices (laptops and computers). She is trained mainly from FAQ datasets extracted from Ministry of Health, Ministry of Education and Immigration and Customs Authority websites. While the main purpose of Misha is to provide answers to user queries, she can respond briefly to users’ small chat as well. 

# Folder Structure
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
    |                     ^- codes to train SBERT model and save trained results in 
    |                       `SBERT/bot` folder
    ├── data/               <-  Folders containing text datasets for training sentence BERT and
    |                       classification model
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

** Download classification model weights/train Sentence BERT model**
5. If you would like to run bot without any voice annotation, type (`python -m chatbot`). To run bot with voice annotation, type (`python -m chatbot_tts`)

# Data Preparation

There are a total of 6 datasets used to train Misha. 

For the Question-Answer feature, FAQs were extracted from Ministry of Health, Immigration and Customs Authority and Ministry of Education websites. The number of question-answer pairs in each dataset were 50, 52 and 139 respectively. As the dataset sizes were small, each question was paraphrased to generate more questions of the same type. A list of common keywords were extracted for each question too. A total of 216 unique keywords were identified, which were then used to identify if a chat input is related to Covid-19.

To train the classifer model to determine if a user input is a question, the SMS dataset and the Stanford Question and Answer Dataset (SQuAD) dataset were used. The SMS dataset contained SMSes from Singapore. During pre-training, SMS text containing "?" were classified as 1 (question) else 0 (not a question). For SQuAD, the dataset used comes with labels of 1 (question) and 0 (non-question).

A small dataset referenced from Google Dialogflow(TM)'s smalltalk agent was created for the purpose of incorporating chat element into Misha to provide a sense of personal touch and engagement to users so that Misha feels more human-like.

![Fig 1: datasets used for model training](https://github.com/beanlee999/mishabot/blob/cy/assets/datasets.png)

# Model Training

These datasets are used to train 2 models:
- BERT classification model to determine if user input is a question or not
- Sentence BERT model to embed user input into a sentence embedding. Dataset used to trained this model is `data/SBERT_train.csv`, which contains a list of questions, question categories and the response to return. 

## Classification Model

## SBERT Model





