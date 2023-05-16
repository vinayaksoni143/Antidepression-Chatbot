import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('database/question_patterns.json', 'r') as json_data:
    intents = json.load(json_data)

with open('database/shloks_output.json', 'r') as shloks_data:
    shloks = json.load(shloks_data)

with open('database/answer_tag_of_shlok.json', 'r') as shloks_mapper:
    shloks_mapper = json.load(shloks_mapper)

FILE = "database/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

def chat_eval(sentence):
    output_statement=''
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag=="greeting":
                    return random.choice(intent['answer'])
                z=shloks_mapper[tag]
                output_statement+='Shlok :'+'\n'+random.choice(shloks[z])+'\n'+'Explanation :'+'\n'+random.choice(intent['answer'])
    else:
        output_statement="I don't understand..."
    return output_statement