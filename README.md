# Introduction

This is a demo chatbot built in 2021, to demonstrate the ability for an AI model to 


# Folder Structure
```
project_repo
    ├── assets              <- Images used for README.md
    ├── README.md           <-  Explains working principle of the Chatbot
    │                       and how to set it up
    ├── requirements.yml    <-  YAML file containing dependency list for setting up
    │                       conda environment
    ├── chatbot.py          <-  Python scripts to set up Misha Chatbot
    ├── chatbot_tts.py      <-  Python scripts to run Misha Chatbot with voice annotation
    ├── SBERT               <-  Folder containing codes for sentence transformers
    ├── weights             <-  Folder containing weights for classification model
    |                       (`classification.pt`)
    ├── codes               <-  Folders containing codes for training 
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
    ├── .gitignore          <-  File for specifying files or directories
    │                       to be ignored by Git.
```

# Steps to set up
To set up and run the chatbot:

1. In windows, open anaconda prompt. In Mac, open terminal. 

2. Change working directory to chatbot project folder directory ("cd <filepathtofolder>/chatbot_v4")

3. Create a conda environment using `requirements.yml` [`conda env create -f requirements.yml`]. The name of the conda environment is `mishabot`

4. Activate conda env (`conda activate mishabot`)

5. If you would like to run bot without any voice annotation, type (`python -m chatbot`). To run bot with voice annotation, type (`python -m chatbot_tts`)

# Data Preparation

![datasets used for model training](https://github.com/beanlee999/mishabot/blob/cy/assets/datasets.png)

# Model Training
## Classification Model

## SBERT Model





