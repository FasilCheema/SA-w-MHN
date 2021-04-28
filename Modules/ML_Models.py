'''
Authors: Fasil Cheema and Mahmoud Alsaeed 
Purpose: This module is used to define all models as functions here 
'''

import keras
from keras.preprocessing import sequence
from keras.layers import Input, Dense,RepeatVector
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,SimpleRNN,GRU,CuDNNGRU,Reshape
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation, Flatten, UpSampling1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import plot_model
import tensorflow as tf
from keras.models import model_from_json
import pickle as pkl
from sklearn.metrics import accuracy_score
from random import shuffle
from keras.layers import Concatenate,Multiply,Add
from keras.layers import TimeDistributed
from keras.models import Sequential
import os
from keras.layers import Flatten
from keras.layers import Lambda
from keras import backend as K

lstmnode=64

def BiLSTM_self_attention_CNN(input_node,max_len):
    encode=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node) #,weights=[vectors],trainable=False
    #embed_class_tweets=Reshape((1,lstmnode*2))(embed_tweets)

    encoder_LSTM,_,_,_,_=Bidirectional(LSTM(lstmnode, return_sequences=True, return_state=True, activation='tanh'))(encode)
    attention = TimeDistributed(Dense(input_tweet, activation='softmax'))(encoder_LSTM)
    


    embed_tweets=keras.layers.dot([attention,encoder_LSTM],(1,1), normalize=False)
    embed_class_tweets=Conv1D(emb,
                     1,
                     padding='same',
                     activation='relu',
                     strides=1)(embed_tweets)
    embed_class_tweets=Dropout(0.2)(embed_class_tweets)
    embed_class_tweets=GlobalMaxPooling1D()(embed_class_tweets)
    return embed_class_tweets


def BiLSTM_CNN(input_node,max_len):
    encode=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node) #,weights=[vectors],trainable=False
    encoder_LSTM,_,_,_,_=Bidirectional(LSTM(lstmnode, return_sequences=True, return_state=True, activation='tanh'))(encode)
    embed_class_tweets=Conv1D(emb,
                     1,
                     padding='same',
                     activation='relu',
                     strides=1)(encoder_LSTM)
    embed_class_tweets=Dropout(0.2)(embed_class_tweets)
    embed_class_tweets=GlobalMaxPooling1D()(embed_class_tweets)
    return embed_class_tweets


def CNN(input_node,max_len):
    encode_hashtags=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node)    #,weights=[vectors],trainable=False
    encode_hashtags=Conv1D(128,
                         3,
                         padding='same',
                         activation='relu',
                         strides=1)(encode_hashtags)
    encode_hashtags=MaxPooling1D(pool_size=1)(encode_hashtags)
    encode_hashtags=Conv1D(64,
                         3,
                         padding='same',
                         activation='relu',
                         strides=1)(encode_hashtags)
    encode_hashtags=Dropout(0.2)(encode_hashtags)

    encode_hashtags=GlobalMaxPooling1D()(encode_hashtags)
    return encode_hashtags

def BiLSTM(input_node,max_len):
    encode=Embedding(len(vectors), emb,weights=[vectors], input_length=max_len, trainable=True)(input_node) #,weights=[vectors],trainable=False
    encoder_BiLSTM=Bidirectional(LSTM(lstmnode,activation='tanh'))(encode)
    return encoder_BiLSTM

def convert2id(text,word2idx=word2idx):
    ids=[]
    for word in text:
        if word in word2idx:
            ids.append(word2idx[word])
    return ids