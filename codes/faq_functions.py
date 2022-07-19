from scipy import spatial
import numpy as np
import pandas as pd
import os
import random
from sentence_transformers import SentenceTransformer
import codes.classification as QnClassifier
import string

class Bot():
    
    def __init__(self, workdir, username, keywords, 
                 bot_SBERT, question_list, conv_list):
        self.botname = "Misha"
        self.username = username
        self.keywords = keywords
        self.qnclassifier = QnClassifier.Classifier(workdir) 
        self.faq = FAQ(self.username, bot_SBERT, question_list[2],
                       question_list[1], question_list[0], 
                       question_list[3])
        self.conversation = Conversation(self.username, bot_SBERT, conv_list[2],
                                         conv_list[1], conv_list[0],
                                         conv_list[3])

    def greetings(self):
        print(f"""
{self.botname::<10}Hi {self.username}, what you would you like to know? I am trained to answer your queries for the below topics:
                              
- MOH Covid 19 guidelines
- MOE Covid 19 guidelines
- ICA Covid 19 immigration guidelines
                              
You can ask me any question and I will try my best to answer you! If you would like to end this conversation at any time, just reply 'end this'\n""")
    
    def userinput(self):
        # status = -2: end conversation, 1: repeat user question, 
        # 0: user usedproceed to next question, 4: exit from smalltalk
        # 3: 
        status = 0
        count = 0
        response = input(f"""{self.username::<10}""")
        if response == 'end this':
            Bot.endconversation(self)
            status = -2
        else:            
            for keyword in self.keywords:
                if response.lower().find(keyword) !=-1 :
                    count = 1
                    break

            if count > 0:
                category = self.qnclassifier.classify(response)
                if category == 1:
                    status = self.faq.question(response)
                else:
                    status = 2
            else:
                status = 3
                status = self.conversation.smalltalk(response)

        return status
    
    def clean_words(self, response):
        
        punctuation = string.punctuation
        response = [t for t in response if t not in punctuation]
        return response.lower()
    
    def endconversation(self):
        print(f"""
{self.botname::<10}

Alright {self.username}, it's sad to end our conversation here. I hope i have been a great help to you. Anyway, I hope you stay safe at at times. Remember to abide by the Safe Management Measures. Wear your mask well and keep sanitising your hands. 
Thank you and good Bye!""")
            
    def failed_response(self):
        print(random.choice([f"{self.botname::<10}Apologies, i did not get your response. Can you repeat again?",
                              f"{self.botname::<10}Sorry! can you repeat again?"]))
        return(input(f"""{self.username::<15}"""))
    

class Conversation:
    """contains functions to run FAQ queries"""
    def __init__(self, username, conv_SBERT, conv_embed,
                 conv_num, conv_text, conv_answer):
        self.botname = "Misha"
        self.username = username
        self.conv_SBERT = conv_SBERT
        self.conv_embed = conv_embed
        self.conv_num = conv_num
        self.conv_text = conv_text
        self.conv_answer = conv_answer

    def smalltalk(self, response):
        encoded_response = self.conv_SBERT.encode([response])
        
        distances = spatial.distance.cdist(np.array(encoded_response), 
                                           self.conv_embed, 'cosine')[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key = lambda x: x[1])
        (idx, distance) = results[:1][0]
        #print(distance)
        print(f"{self.botname::<10}{self.conv_answer[idx]}")
        return 3
        

class FAQ:
    """contains functions to run FAQ queries"""
    def __init__(self, username, faq_SBERT, question_embed,
                 question_num, question_text, question_answer):
        self.botname = "Misha"
        self.faq_SBERT = faq_SBERT
        self.question_embed = question_embed
        self.question_num = question_num
        self.question_text = question_text
        self.question_answer = question_answer
        self.username = username
    
    def question(self, new_question):
        # new_question = input(random.choice([f"{self.botname::<10}Hi {self.username}, Sure! What question would you like to ask?\n{self.username::<10}",
        #                                     f"{self.botname::<10}Hi {self.username}, what would you like to know?\n{self.username::<10}"]))
        
        idx, distance, reply = FAQ.checkifsimilar(self, new_question)
        reply = FAQ.check(self, reply)
        if reply == 1:
            print()
            print(random.choice([f"{self.botname::<10}Here's the information that you are looking for.",
                                 f"{self.botname::<10}This is the information that you need.",
                                 f"{self.botname::<10}Here is the answer that you wanted."]))
            print(f"'\x1B[3m{self.question_answer[idx]}\x1B[0m'\n")
            return 0
        elif reply == -2:
            Bot.endconversation(self)
            return -2
        else:
            return 1

    def checkifsimilar(self, new_question):
        encoded_question = self.faq_SBERT.encode([new_question])
        
        distances = spatial.distance.cdist(np.array(encoded_question), 
                                           self.question_embed, 'cosine')[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key = lambda x: x[1])
        (idx, distance) = results[:1][0]
        
        print(f"{self.botname::<10}I found a question similar to what you asked.")
        print(f"'\x1B[3m{self.question_text[idx]}\x1B[0m'\n")
        #\x1B[3mitalic text\x1B[0m
        
        reply = input(random.choice([f"{self.botname::<10}Is this what you are looking for?\n{self.username::<10}",
                                     f"{self.botname::<10}Are you looking for this?\n{self.username::<10}",
                                     f"{self.botname::<10}Is this what you wanted to know?\n{self.username::<10}"]))
        return idx, distance, reply
    
    def check(self, reply):
        pos = ['ya', 'yes', 'yup', 'correct', 'right', 'bingo', 'great', 'ok']
        neg = ['no', 'nope', 'wrong', 'negative', 'incorrect']
        if reply.lower() == 'end this':
            return -2
        for word in reply.lower().split(' '):
            if word in pos:
                return 1
            elif word in neg:
                return -1
            else:
                return 0

    def failed_response(self):
        print(random.choice([f"{self.botname::<10}Apologies, I did not get your response, can you repeat again? ",
                             f"{self.botname::<10}Sorry! Do you mind saying again?"]))
        return(input(f"""{self.username::<10}"""))

