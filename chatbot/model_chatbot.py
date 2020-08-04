

import nltk
from nltk.stem import WordNetLemmatizer
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub

lemmatizer = WordNetLemmatizer()
words=[]
documents = []
ignore_words = ['?', '!', '.', ',']
data_file = open('Intents2.json').read()
intents = json.loads(data_file)

sentences_p =[]

for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        sentences_p.append(pattern)
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))


# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


y = []
for item in documents:
    y.append(item[1])
        

vocab_size = len(words)
max_length = 20

lower_final_sentences = []
for snt  in sentences_p:
    lower_final_sentences.append(snt.lower())
    

#BERT 
from bert_embedding import BertEmbedding

import bert
import numpy as np
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


bert_group =[]
for snt in lower_final_sentences:
    bert_group.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(snt)))


train_x = pad_sequences(bert_group, maxlen=max_length,padding='post')
vocab_size=len(tokenizer.vocab)

# ONE HOT ENCODING
#q =[one_hot(d, vocab_size) for d in lower_final_sentences]
#train_x = pad_sequences(q, maxlen=max_length,padding='post')


#Label Encoding for y
encoder = LabelEncoder()
y = encoder.fit_transform(y)





#Create the model
from tensorflow.keras import layers
model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length,name="emd1"))

model.add(Dropout(0.2,name="drp1"))
model.add(LSTM(100,name="lsm"))
model.add(Dropout(0.2,name="drp2"))
model.add(Dense(15, activation='sigmoid',name="dense1"))

model.summary()

#Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Fitting and saving the model 
hist = model.fit(train_x, y,
                 validation_split=0.10, epochs=400, batch_size=5, verbose=2)

loss, accuracy = model.evaluate(train_x,y)
print(loss)
print(accuracy)


#model.add(layers.GRU(100, return_sequences=True))
#model.add(layers.SimpleRNN(128))   
    
model.save('chatbot_Bert_9090.h5', hist)







