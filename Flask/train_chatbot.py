import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import random
import numpy as np
import random
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub



words=[]
classes = []
documents = []
sentences_p =[]
ignore_words = ['?', '!', '.', ',']
data_file = open('Intents2.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        sentences_p.append(pattern)
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

resp_array = []
for intent in intents['intents']:
    for res in intent['responses']:
        resp_array.append(res)
    


#import video_emotion_color_demo as file
emotion_tag = "happy"

max_length = 20
vocab_size = len(words)


y = []
for item in documents:
    y.append(item[1])
        
lower_final_sentences = []
for snt  in sentences_p:
    lower_final_sentences.append(snt.lower())
    
#from bert_embedding import BertEmbedding




intents = json.loads(open('Intents2.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


import bert

BertTokenizer = bert.bert_tokenization.FullTokenizer


bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)


vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()


to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()


tokenizer = BertTokenizer(vocabulary_file, to_lower_case)



def chat(word,model):

        inp = word

        vocab_size = len(tokenizer.vocab)
        l =[]
        inp = inp.lower()
        l.append(inp)
        #ONE HOT ENCODER
        #v = [one_hot(d, vocab_size) for d in l]
        #BERT
        user = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inp))]
        padded_docs = pad_sequences(user,max_length,padding='post')
        results = model.predict(padded_docs)
        result_index = np.argmax(results)
        tag = classes[result_index]
        print(tag)
        
        for t in intents['intents']:
            if t['tag'] == tag:
                responses = t['responses']
                
        return random.choice(responses)
                
            
        
        








