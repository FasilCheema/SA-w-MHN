from GraphGenerator import GraphGenerator
from Preprocessor import Preprocesso
from ML_models import *
from Embedder import * 



f_tweet = open('/home/fasil/Repos/SA_Hetero_Net/data/semeval13_multiplex_network.pkl', 'rb')
tweet_net=pickle.load(f_tweet)     #((hashtags,mentions,edges,h_k,m_k),res,label)
f_tweet.close()


f_tweet = open('/home/fasil/Repos/SA_Hetero_Net/data/semeval13_biased_rw_startnode_without_expansion_lm.pkl', 'rb')  
preprocess_corpus_startnode_id=pickle.load(f_tweet)     #((hashtags,mentions,edges,h_k,m_k),res,label)  
f_tweet.close() 


num_walk=30
max_len=30

training_ips=[]
testing_ips=[]
training=[]
testing=[]
training_rw=[]
testing_rw=[]
y_test=[]
y_train=[]
no_rw=3
test_tids=[]
for i in range(no_rw):
    training.append([])
    testing.append([])
    training_rw.append([])
    testing_rw.append([])

for tid in tweet_net:
    if 'train' in tid and tid in preprocess_corpus_startnode_id and len(preprocess_corpus_startnode_id[tid])>=no_rw:
        samples=[]
        for q1 in tweet_net[tid][1]:
            q1=q1.replace('@','M_').replace('#','H_')
            if q1 in word2idx:  #q1 not in stopwords and 
                samples.append(word2idx[q1])
        training_ips.append(samples)
        for i in range(no_rw):
            training_rw[i].append(convert2id(preprocess_corpus_startnode_id[tid][i][0],word2idx))
        y_train.append(tweet_net[tid][2])
        # for i in range(no_rw):
        #     training[i].append(d2vec[tid][i])     


# for tid in test_tid:
    if 'test' in tid and tid in preprocess_corpus_startnode_id and len(preprocess_corpus_startnode_id[tid])>=no_rw:
        test_tids.append(tid)
        samples=[]
        for q1 in tweet_net[tid][1]:
            q1=q1.replace('@','M_').replace('#','H_')
            if q1 in word2idx:  #q1 not in stopwords and 
                samples.append(word2idx[q1])
        testing_ips.append(samples)
        for i in range(no_rw):
            testing_rw[i].append(convert2id(preprocess_corpus_startnode_id[tid][i][0],word2idx))
        y_test.append(tweet_net[tid][2])



training_ips = sequence.pad_sequences(training_ips, maxlen=max_len)
testing_ips = sequence.pad_sequences(testing_ips, maxlen=max_len)

for i in range(no_rw):
    training_rw[i]=sequence.pad_sequences(training_rw[i], maxlen=max_len)
    testing_rw[i]=sequence.pad_sequences(testing_rw[i], maxlen=max_len)

training1=[training_ips]+training_rw
testing1=[testing_ips]+testing_rw

y_train=convert_labels(y_train)
y_test=convert_labels(y_test)

y_train = np.array(y_train)
y_test = np.array(y_test)



del preprocess_corpus_startnode_id
gc.collect()

emb=len(vectors[0])

input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=CNN(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(CNN(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)



embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))



model_json = model.to_json()
with open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_CNN_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_CNN_ensemble.h5")
print("Saved model to disk")      

result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_CNN_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()


input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=BiLSTM_CNN(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(BiLSTM_CNN(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)

embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))


model_json = model.to_json()
with open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_CNN_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_CNN_ensemble.h5")
print("Saved model to disk")      



result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_CNN_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()



input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=BiLSTM(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(BiLSTM(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)



embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))



model_json = model.to_json()
with open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_ensemble.h5")
print("Saved model to disk")      



result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()



input_tweet=len(training1[0][0])
input_node=Input(shape=(input_tweet,))
emb_CNN=BiLSTM_self_attention_CNN(input_node,input_tweet)
# emb_CNN=Dropout(0.2)(emb_CNN)

input_rw_node=[]
emb_rw=[]
input_rw=len(training1[1][0])

for i in range(no_rw):
    input_rw_node.append(Input(shape=(input_rw,)))
    emb_rw.append(BiLSTM_self_attention_CNN(input_rw_node[i],input_rw))

rw_tweets=Concatenate()(emb_rw)
embed_rw=Dense(60, activation='relu')(rw_tweets)
# embed_rw=Dropout(0.2)(embed_rw)



embed_hash_tweets=Concatenate()([emb_CNN,embed_rw])
# embed_hash_tweets=Dropout(0.2)(embed_hash_tweets)
senti_class=Dense(3,activation='softmax')(embed_hash_tweets)
model=Model([input_node]+input_rw_node,senti_class)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training1, y_train,
          batch_size=512,
          epochs=10,
          validation_data=(testing1, y_test))


model_json = model.to_json()
with open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_self_attention_CNN_ensemble.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_self_attention_CNN_ensemble.h5")
print("Saved model to disk")      



result_LSTM=[]
result_LSTM=model.predict(testing1,verbose=0)
fd = open("/home/fasil/Repos/SA_Hetero_Net/model/semeval13_biased_rw_ft_emb_no_expansion_"+str(no_rw)+"_BiLSTM_self_attention_CNN_ensemble","w")
for ii,i in enumerate(result_LSTM):
    fd.write(test_tids[ii]+'\t')
    if y_test[ii][0] == 1:
        fd.write('Positive\t')
    elif y_test[ii][1] == 1:
        fd.write('Negative\t')
    else:
        fd.write('Neutral\t')
        
    if i[0] > i[1]:
        if i[0] > i[2]:
            fd.write('Positive\n')
        else:
            fd.write('Neutral\n')
    elif i[1] > i[2]:
        fd.write('Negative\n')
    else:
        fd.write('Neutral\n')
fd.close()
