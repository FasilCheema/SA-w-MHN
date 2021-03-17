'''
Authors: Fasil Cheema and Mahmoud Alsaeed
Purpose: This module is used to create a multilayer network represented as a graph with 3 layers; keywords, hashtags, and mentions. 
'''

import networkx as nx 
import numpy as np 

class GraphGenerator(text_data, label_data):
    # initialize class
    def __init__(self):
        self.layer_words = ["#","@"]

    def StringParser(self):
        # Take text data and feed into next module sequentially 

    def MatrixGenerator(self, tweet):
        # Create the necessary adjacency matrices as described in paper
        hashtag_pos_list = []
        mention_pos_list = []

        list_tweet = tweet.split()

        # Searches for hashtags and mentions and keeps track of their location in the tweet
        for curr_word in list_tweet:
            if curr_word[0] == "#":
                hashtag_pos_list.append(list_tweet.index(curr_word))

                if curr_word[0]
            elif curr_word[0] == "@":
                mention_pos_list.append(list_tweet.index(curr_word))
        
        num_hashtag = len(hashtag_pos_list) 
        num_mention = len(mention_pos_list)
        num_keyword = len(list_tweet) - (num_hashtag + num_mention)

        # Initialize matrices
        A_h = np.ones((num_hashtag,num_hashtag))
        A_m = np.ones((num_mention,num_mention))
        A_k = np.zeros((num_keyword,num_keyword))
        
        B_hk = np.zeros((num_hashtag,num_keyword))
        B_mk = np.zeros((num_mention,num_keyword))
        
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

        # Create intra-layer adjacency matrix for keyword layer
        anchor_list = hashtag_pos_list + mention_pos_list
        anchor_list = sorted(anchor_list)

        for index_val in anchor_list:
            





         
        


if __name__ == "__main__":
    test_tweet = "@handle word word word word #hashtag @mention word word #hashtag"
    test_label = 
    