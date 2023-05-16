
import numpy as np
import nltk 
import string 
import random
from nltk.corpus import stopwords
from nltk_utils import  tokenize

nltk.download('punkt')
nltk.download('wordnet')

f=open('Shloks.txt','r',errors='ignore')
raw_doc=f.read()
raw_doc=raw_doc.strip()
raw_doc=raw_doc.lower()
sent_tokens=nltk.sent_tokenize(raw_doc)
word_tokens= nltk.word_tokenize(raw_doc)

lemmer=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]

remove_punt_dict = dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punt_dict)))

#Greeting 
GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREET_RESPONSES = ["hi", "hey", "*nods", "hi there", "hello", "I am glad! You are talking to me"] 

def greet (sentence):
  for word in sentence.split():
    if word. lower() in GREET_INPUTS:
       return random.choice (GREET_RESPONSES)

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
 
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

def preprocessing(user_response):
  # Tokenize and tag
  sentence2 = pos_tag(word_tokenize(user_response))
  # Get the synsets for the tagged words
  synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
  # Filter out the Nones
  synsets2 = [ss for ss in synsets2 if ss]
  return synsets2
  
  
def similarity_ratio(user_response):
  sentence1="I would like to keep talking"
  sentence1 = pos_tag(word_tokenize(sentence1))

  sentence3="I want a solution "
  sentence3 = pos_tag(word_tokenize(sentence3))


   
  synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
  synsets3 = [tagged_to_synset(*tagged_word) for tagged_word in sentence3]
 
   
  synsets1 = [ss for ss in synsets1 if ss]
  synsets3 = [ss for ss in synsets3 if ss]
  synsets2= preprocessing(user_response)


    # Score calculations for continuing or wanting a solution
    #wrt synset 1
  score1, count1 = 0.0, 0
  for synset in synsets1:
    best_score1 = max([synset.path_similarity(ss) for ss in synsets2])
    if best_score1 is not None:
      score1 += best_score1
      count1 += 1
  score1/=count1

  score3, count3 = 0.0, 0
  for synset in synsets3:
    best_score3 = max([synset.path_similarity(ss) for ss in synsets2])
    if best_score3 is not None:
      score3 += best_score3
      count3 += 1
  score3/=count3

  return score1,score3

#In case user wants to continue 

CONTINUE_RESPONSES = ["You are doing great keep going!", "I see, then ?", "*nods in understanding*", "So then what happened next", "Please continue, I'm all ears", "I am glad! You are talking to me"] 

def continue_talk():
       return random.choice (CONTINUE_RESPONSES)

# In case user wants a solution, Tanishq's model will come here 

def want_solution():

# Response Generation
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
  robo1_response=' '
  TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
  tfidf= TfidfVec.fit_transform(sent_tokens)
  vals = cosine_similarity(tfidf[-1],tfidf)
  idx=vals.argsort()[0][-2]
  flat=vals.flatten()
  flat.sort()
  req_tfidf=flat[-2]
  if(req_tfidf==0):
    robo1_response=robo1_response+"I am sorry! Didn't quite get what you said."
    return robo1_response
  else:
    robo1_response=robo1_response+sent_tokens[idx]
    return robo1_response

flag=True
print("BOT: Hello my name is Radha. Let's have a conversation! Also if you want to exit any time, just type Bye!")
while(flag==True):
  user_response = input()
  user_response=user_response.lower()
  score_continue,score_sol= similarity_ratio(user_response)
  if (user_response!= 'bye'):
    if(user_response==' thanks' or user_response=='thank you' ):
      flag=False
      print("BOT: You are welcome..")

    else:
      if(greet (user_response)!=None): 
        print("BOT: "+greet (user_response))

      else:
        sent_tokens.append(user_response)
        word_tokens=word_tokens+nltk.word_tokenize(user_response)
        final_words=list(set(word_tokens))
        print("BOT: ",end="")
        print(response(user_response))
        sent_tokens.remove(user_response)

    if (score_continue>score_sol):
      print("BOT: " + continue_talk())

    #if (score_sol>score_continue):
      #print("BOT: " + want_solution())


  else:
    flag=False
    print("BOT: Goodbye! Take care <3")

"""## This is rough work"""



import spacy
from spacy.lang.en import English

# Define the training data
TRAIN_DATA = [
    ("Hello, how are you?", {"greeting": True}),
    ("Hey, let's keep talking.", {"trying_to_continue": True}),
    ("Can you help me with this?", {"asking_for_solution": True}),
    ("I think we're done here.", {"ending_conversation": True}),
    ("What is your favorite color?", {"other": True}),
    # Add more examples for each class
]

# Train the intent classifier
def train_intent_classifier(nlp):
    # Create the TextCategorizer pipeline component
    textcat = nlp.create_pipe("textcat_multilabel")

    # Add the labels to the text classifier
    for label in ["greeting", "trying_to_continue", "asking_for_solution", "ending_conversation", "other"]:
        textcat.add_label(label)

    # Add the TextCategorizer component to the pipeline
    nlp.add_pipe('textcat', last=True)

    # Disable other pipeline components for training
    pipe_exceptions = ["textcat_multilabel", "tokenizer"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*unaffected_pipes):
        # Training parameters
        optimizer = nlp.begin_training()

        # Iterate through training data
        for i in range(10):  # Adjust the number of iterations as needed
            for text, labels in TRAIN_DATA:
                doc = nlp.make_doc(text)
                gold = spacy.gold.GoldParse(doc, cats=labels)
                nlp.update([doc], [gold], sgd=optimizer)

# Perform intent classification
def classify_intent(nlp, text):
    doc = nlp(text)
    intent = doc.cats
    classified_intent = [label for label, score in intent.items() if score >= 0.5]
    return classified_intent[0] if classified_intent else "other"

# Load spaCy model and train intent classifier
nlp = English()
train_intent_classifier(nlp)

# Test intent classification
text_inputs = [
    "Hello, how's it going?",
    "Let's continue our conversation.",
    "I need help with a problem.",
    "I think we can end the conversation now.",
    "This is some random text."
]

for text in text_inputs:
    intent = classify_intent(nlp, text)
    print(f"Text: {text}\nIntent: {intent}\n")

from spacy.gold import GoldParse



