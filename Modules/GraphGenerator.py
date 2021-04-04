'''
Authors: Fasil Cheema and Mahmoud Alsaeed
Purpose: This module is used to create a multilayer network represented as a graph with 3 layers; keywords, hashtags, and mentions. 
'''

import networkx as nx 
import numpy as np 

class GraphGenerator():
    # initialize class
    def __init__(self):
        self.layer_words = ["#","@"]

    def StringParser(self):
        # Take text data and feed into next module sequentially 
        return null

    def MatrixGenerator(self, tweet):
        # Create the necessary adjacency matrices as described in paper
        hashtag_pos_list = []
        mention_pos_list = []
        layer_val_list   = []

        list_tweet = tweet.split()
        

        # Searches for hashtags and mentions and keeps track of their location in the tweet
        for curr_word in list_tweet:
            if curr_word[0] == "#":
                hashtag_pos_list.append(list_tweet.index(curr_word))
                layer_val_list.append(1)
            elif curr_word[0] == "@":
                mention_pos_list.append(list_tweet.index(curr_word))
                layer_val_list.append(2)
            else:
                layer_val_list.append(0)


        num_hashtag = len(hashtag_pos_list) 
        num_mention = len(mention_pos_list)
        num_keyword = len(list_tweet) - (num_hashtag + num_mention)

        # Initialize matrices
        A_h = np.ones((num_hashtag,num_hashtag))
        A_m = np.ones((num_mention,num_mention))
        A_k = np.zeros((num_keyword,num_keyword))
        
        B_hk = np.zeros((num_hashtag,num_keyword))
        B_mk = np.zeros((num_mention,num_keyword))
        B_kh = np.transpose(B_hk)
        B_km = np.transpose(B_mk)
        
        # Create intra-layer adjacency matrix for hashtag layer by setting diagonal to zero
        for i in range(num_hashtag):
            A_h[i,i] = 0


        # Create intra-layer adjacency matrix for mention layer by setting diagonal to zero
        for i in range(num_mention):
            A_m[i,i] = 0
        
        """Note: as described in paper we are not creating the adjacency matrix
                 for bipartite graphs, but the biadjacency matrix, the unique matrix that fully describes the bipartite graph  """
        # Create bipartite matrix for mention and hashtag layer
        B_hm = np.ones((num_hashtag,num_mention))
        B_mh = np.transpose(B_hm)


            
        for k in range(len(layer_val_list)-1):
            
            curr_val = layer_val_list[k]
            next_val = layer_val_list[k+1]

            if curr_val == 0 and next_val == 0:
                A_k[k,k+1] = 1
            elif curr_val == 0 and next_val == 1:
                B_kh[k,k+1] = 1
            elif curr_val == 1 and next_val == 0:
                B_hk[k,k+1] = 1
            elif curr_val == 0 and next_val == 2:
                B_km[k,k+1] = 1
            elif curr_val == 2 and next_val == 0:
                B_mk[k,k+1] = 1

        return A_h, A_k, A_m, B_hk, B_hm, B_kh, B_km, B_mh, B_mk