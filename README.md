**Introduction**

This is a demo chatbot built in 2021, to demonstrate the ability for an AI model to 


**Folder Structure**
```
project_repo
    ├── assets              <- Images used for README.md
    ├── README.md           <-  Explains working principle of the Chatbot
    │                       and how to set it up
    ├── requirements.yml    <-  YAML file containing dependency list for setting up
    │                       conda environment
    ├── .dockerignore       <-  File for specifying files or directories
    │                       to be ignored by Docker contexts.
    ├── .pylintrc           <-  Configurations for `pylint`.
    ├── .gitignore          <-  File for specifying files or directories
    │                       to be ignored by Git.
    ├── data                <-  Folders containing images files for model training
    │   ├── cleaned         <-  Directory containing raw zipped folders for 
    |   |                   training image
    │   ├── extracted       <-  Directory containing extracted unzipped folders for 
    |                       training images, and image metadata stored in .pkl file

```

**Steps to set up**
To set up and run the chatbot:

1. In windows, open anaconda prompt. In Mac, open terminal. 

2. Change working directory to chatbot project folder directory ("cd <filepathtofolder>/chatbot_v4")

3. Create a conda environment using `requirements.yml` [`conda env create -f requirements.yml`]. The name of the conda environment is `mishabot`

4. Activate conda env (`conda activate mishabot`)

5. If you would like to run bot without any voice annotation, type (`python chatbot.py`). To run bot with voice annotation, type (`python chatbot_tts.py`)



