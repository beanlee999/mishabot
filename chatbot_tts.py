#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:58:33 2021

@author: Team FLLY NUS ISS MTech EBAC 2020-2021

"""

import pandas as pd
import os
import random
import codes.faq_functions_tts as faq_functions
from sentence_transformers import SentenceTransformer
import pyttsx3

engine = pyttsx3.init()

#%%
engine.setProperty('rate', 175)     # setting up new voice rate
engine.runAndWait()
voices = engine.getProperty('voices')       # getting details of current voice
engine.setProperty('voice', voices[28].id)  # Use 10, 17 or 28
engine.runAndWait()

#%%

# Preparing FAQ lists
workdir = input("Please input chatbot project directory: ")

print('Please wait a moment while the chatbot is being set up .........')

# Load SBERT
bot_SBERT = SentenceTransformer(os.path.join(workdir, 'SBERT/bot'))
# faq_SBERT = SentenceTransformer(os.path.join(workdir, 'SBERT/FAQ'))
# conv_SBERT = SentenceTransformer(os.path.join(workdir, 'SBERT/smalltalk'))

#%%
faq = pd.read_csv(os.path.join(workdir, 
                                'data/FAQs.csv'),
                    encoding = "ISO-8859-1")

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


#%%
# Load small talk SBERT
conv = pd.read_csv(os.path.join(workdir, 'data/small_talk.csv'))

conv_text = list(conv.iloc[:,0])
conv_num = list(conv.iloc[:,1])
conv_embed = []
for line in conv_text:
    conv_embed.append(bot_SBERT.encode(line))
conv_answer = list(conv.iloc[:,2])
conv_list = [conv_text, conv_num, conv_embed, conv_answer]

#%%
# Chatbot function
botname = "Misha"
print(f"""{botname::<10}""")
text = """
Good day! My name is Misha. 
I am a bot developed to answer your queries on Covid-19 guidelines in Singapore. 
Before we start, what is your name please?\n\n"""
print(text)
engine.say(text)
engine.runAndWait()
username = input(f"""{"User"::<10}""")


# Create an instance of bot
bot = faq_functions.Bot(workdir, username, keywords, 
                        bot_SBERT, question_list,conv_list)


# Start Bot
# intent = intent_classification.Sentiment(workdir)
# intent.output('hello')

# conversation = faq_functions.Conversation(username, conv_SBERT, conv_embed,
#                                           conv_num, conv_text, conv_answer)

bot.greetings()
while True:
    status = bot.userinput()
    if status == -2:
        break
    elif status == 0:
        text = random.choice([f"Hi {username}, how else may I help you with?",
                              f"Hi {username}, what else would you like to know?",
                              f"Hi {username}, you may continue to ask me other questions."])
        print(f"{botname::<10}{text}")
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
        engine.say(text)
        engine.runAndWait()
    
    elif status == 1:
        text = random.choice(["Ok, apologies. Can you repeat your question again?",
                              "Ok, apologies. Would you mind repeating your question again?"])
        print(f"{botname::<10}{text}")
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
        engine.say(text)
        engine.runAndWait()   
    
    elif status == 2:
        text = random.choice(["I sense that you are keen in topics that I was trained in. Would you like to know anything?",
                              "You just talked about something I am familiar with. Would you like to know anything?"])
        print(f"{botname::<10}{text}")
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
        engine.say(text)
        engine.runAndWait()     
    
    elif status == 3:
        text = random.choice(["Would you like to ask me anything?",
                              "Can I help you with anything?",
                              "Would you like to know anything?",
                              "How may i help you with?",
                              "Is there anything that you would like to know?"])
        print(f"{botname::<10}{text}")
        print("'\x1B[3m(Enter 'end this' to end this conversation)\x1B[0m'\n")
        engine.say(text)
        engine.runAndWait() 

    else:
        continue
        


