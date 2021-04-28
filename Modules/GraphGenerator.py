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

    def MatrixGenerator(self, tweet):
        # Create the necessary adjacency matrices as described in paper
        
        #Keeps position of each relevant layers in the tweet
        hashtag_pos_list = []
        mention_pos_list = []
        keyword_pos_list = []
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
                keyword_pos_list.append(list_tweet.index(curr_word))
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
        B_kh = np.zeros((num_keyword,num_hashtag))
        B_km = np.zeros((num_keyword,num_mention))
        
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
                k_pos1 = keyword_pos_list.index(k)
                k_pos2 = keyword_pos_list.index(k+1)
                A_k[k_pos1,k_pos2] = 1
            elif curr_val == 0 and next_val == 1:
                k_pos = keyword_pos_list.index(k)
                h_pos = hashtag_pos_list.index(k+1)
                B_kh[k_pos,h_pos] = 1
            elif curr_val == 1 and next_val == 0:
                h_pos = hashtag_pos_list.index(k)
                k_pos = keyword_pos_list.index(k+1)
                B_hk[h_pos,k_pos] = 1
            elif curr_val == 0 and next_val == 2:
                k_pos = keyword_pos_list.index(k)
                m_pos = mention_pos_list.index(k+1)
                B_km[k_pos,m_pos] = 1
            elif curr_val == 2 and next_val == 0:
                m_pos = mention_pos_list.index(k)
                k_pos = keyword_pos_list.index(k+1)
                B_mk[m_pos,k_pos] = 1

        return A_h, A_k, A_m, B_hk, B_hm, B_kh, B_km, B_mh, B_mk

    def SupraMatrixGenerator(self, A_h, A_k, A_m, B_hk, B_hm, B_kh, B_km, B_mh, B_mk):
        ''' Creates Supra Adjacency matrix as described in the paper, using the relevant given matrices the function basically assembles it
        '''
        #First concatenate along the row axes, take three of the matrices and join them to make one giant column matrix 
        row_mat1 = np.concatenate((A_h,B_mh,B_kh))
        row_mat2 = np.concatenate((B_hm,A_m,B_km))
        row_mat3 = np.concatenate((B_hk,B_mk,A_k))

        #From the previous block, take the giant column matrices and now stack them horizontally to create the supra matrix
        supra_matrix = np.concatenate((row_mat1,row_mat2,row_mat3),axis=1)

        return supra_matrix
    def ProbMatrixGenerator(self, A_h, A_k, A_m, B_hk, B_hm, B_kh, B_km, B_mh, B_mk, lambda_param, delta, num_layers=3):
        ''' Takes in 9 special matrices and modifies them to create a probability transition matrix that is necessary to create random walks.
            More specifically it alters the matrices to create 9 new matrices that are then fed into the Supra Matrix generator function 
            defined in this class.
        '''
        Ah_p  = (1-lambda_param)*A_h
        Ak_p  = (1-lambda_param)*A_k
        Am_p  = (1-lambda_param)*A_m
        Bhk_p = (lambda_param/(num_layers-1))*B_hk
        Bhm_p = (lambda_param/(num_layers-1))*B_hm
        Bkh_p = (lambda_param/(num_layers-1))*B_kh
        Bkm_p = (lambda_param/(num_layers-1))*B_km
        Bmh_p = (lambda_param/(num_layers-1))*B_mh
        Bmk_p = (lambda_param/(num_layers-1))*B_mk

        return Ah_p, Ak_p, Am_p, Bhk_p, Bhm_p, Bkh_p, Bkm_p, Bmh_p, Bmk_p
