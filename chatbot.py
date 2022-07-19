from scipy import spatial
import numpy as np
import pandas as pd
import os
import random
import codes.faq_functions as faq_functions
from sentence_transformers import SentenceTransformer

# Preparing FAQ lists
workdir = os.getcwd()

print('Please wait a moment while the chatbot is being set up .........')

# Load SBERT
bot_SBERT = SentenceTransformer(os.path.join(workdir, 'SBERT//bot'))
# faq_SBERT = SentenceTransformer(os.path.join(workdir, 'SBERT/FAQ'))
# conv_SBERT = SentenceTransformer(os.path.join(workdir, 'SBERT/smalltalk'))

faq = pd.read_csv(os.path.join(workdir, 'data/FAQs.csv'), encoding = "ISO-8859-1")

# Encode FAQs
question_text = list(faq.iloc[:,0])
question_num = list(faq.iloc[:,1])
question_embed = []
for question in question_text:
    question_embed.append(bot_SBERT.encode(question))
question_answer = list(faq.iloc[:,2])

question_list = [question_text, question_num, question_embed, question_answer]

# Prepare key words from FAQs
keywords = []
for i in range(len(faq)):
    try:
        row = faq[['keywords']].iloc[i].values[0].lower().split(', ')
        keywords.extend(row)
    except:
        pass
keywords = sorted(set(keywords))


# Load small talk SBERT
conv = pd.read_csv(os.path.join(workdir, 'data/small_talk.csv'))

conv_text = list(conv.iloc[:,0])
conv_num = list(conv.iloc[:,1])
conv_embed = []
for line in conv_text:
    conv_embed.append(bot_SBERT.encode(line))
conv_answer = list(conv.iloc[:,2])
conv_list = [conv_text, conv_num, conv_embed, conv_answer]


# Chatbot function
botname = "Misha"
username = input(
f"""{botname::<10}

Good day! My name is Misha. 
I am a bot developed to answer your queries on Covid-19 guidelines in Singapore. 
Before we start, what is your name please?\n\n{"User"::<10}""")

# Create an instance of bot
bot = faq_functions.Bot(workdir, username, keywords, 
                        bot_SBERT, question_list,conv_list)


# Start Bot
bot.greetings()
while True:
    status = bot.userinput()
    if status == -2:
        break
    elif status == 0:
        print(random.choice([f"{botname::<10}Hi {username}, how else may I help you with? You can ask me if you need to",
                             f"{botname::<10}Hi {username}, what else would you like to know? I will try my best to answer you",
                             f"{botname::<10}Hi {username}, you may continue to ask me other questions. I will try to find an answer for you if I can"]))
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
    elif status == 1:
        print(random.choice([f"{botname::<10}Ok, apologies. Can you repeat your question again?",
                             f"{botname::<10}Ok, apologies. Would you mind repeating your question again?"]))
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
    elif status == 2:
        print(random.choice([f"{botname::<10}I sense that you are keen in topics that I was trained in. Would you like to know anything?",
                             f"{botname::<10}You just talked about something I am familiar with. Would you like to know anything?"]))
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
    elif status == 3:
        print(random.choice([f"{botname::<10}Would you like to ask me anything?",
                             f"{botname::<10}Can I help you with anything?"]))
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
    else:
        continue