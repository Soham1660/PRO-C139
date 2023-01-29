# Text Data Preprocessing Lib
import nltk

import random
import json
import pickle
import numpy as np


ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow
model=tensorflow.keras.models.load("./chatbot_model.h5")
from data_preprocessing import get_stem_words
#load data files
intents=json.loads(open("./intents.json").read())
words=pickle.load(open("./words.pkl","rb"))
classes=pickle.load(open("./classes.pkl","rb"))
def preprocess_user_input(user_input):
   input_word_token_1=nltk.word_tokenize(user_input)
   input_word_token_2=get_stem_words(input_word_token_1,ignore_words)
   input_word_token_2=sorted(list(set(input_word_token_2)))
   bag=[]
   bag_of_words=[]
   for word in words:
     if word in input_word_token_2:
        bag_of_words.append(1)
     else:
        bag_of_words.append(0)
   bag.append(bag_of_words)
   return np.array(bag)
def bot_class_prediction(user_input):
    inp=preprocess_user_input(user_input)
    prediction=model.predict(inp)
    predicted_class_label=np.argmax(prediction[0])
    return predicted_class_label        
def bot_response(user_input):
    predicted_class_label=bot_class_prediction(user_input)
    predicted_class=classes[predicted_class_label]
    for intent in intents["intents"]:
        if intent["tag"]==predicted_class:
            return random.choice(intent["responses"])
print("hi I am David, how can I help you with this")
while True:
    user_input=input("What can I help you with? ")
    print("user input", user_input)
    response=bot_response(user_input)
    print("Hello, bot response", response)
