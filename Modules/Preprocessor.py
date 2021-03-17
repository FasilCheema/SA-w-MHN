'''
Authors: Fasil Cheema and Mahmoud Alsaeed
Purpose: This module preprocesses the data (tweets) in preperation to create a network
'''

import os
import pandas as pd
import numpy as np  

class Preprocessor():
    def __init__(self):
        # Initialize File Names 
        self.fname_train = 'tweet_matrix_semeval13_train'
        self.fname_test  = 'tweet_matrix_semeval13_test'

        # Some properties of the preprocesser that can be modified here
        self.elim_list   = [',','.','?'] #list of characters that should be eliminated 
        self.delim_list  = ['@','#'] #list of characters for each layer in network (by default we always use generic words as one)
    def initialize_dir(self):
        # Changes directory to parent directory for accessibility (goes up one directory from modules) 
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)

        return path_parent

    def import_raw_data(self, dir_name):
        data = pd.read_csv(dir_name, sep='\t', header=None)

        return data

    def clean_data(self, data):
        # splits data into instances and labels, then eliminates certain characters (can be modified as a class' property)
        text_data = data.loc[:,0]  #dtype = object
        label = data.loc[:,1]      #dtype = int64
        
        num_pts = len(text_data)

        # initialize an empty dataframe for clean data 
        text_data_clean = pd.DataFrame(np.zeros((num_pts)), dtype = np.int64)
        text_data_clean = text_data_clean[0] # need to be fixed (only 1 column, shape needs to be adjusted)

        for i in range(num_pts):
            text_data_clean[i] = text_data[i].lower()
            for curr_char in self.elim_list:
                text_data_clean[i] = text_data_clean[i].replace(curr_char,"")

        return text_data_clean





if __name__ == '__main__':

    Preprocessor = Preprocessor()

    # Test for initialize_dir
    parent_dir = Preprocessor.initialize_dir()
    print(parent_dir)
    print('end test for initialize_dir')

    # Test for import_raw_data
    new_dir = parent_dir + '/Data/' + Preprocessor.fname_train
    print(new_dir)
    data = Preprocessor.import_raw_data(new_dir)
    print(data.shape)
    print(data.loc[:,0])
    print(data.loc[:,1])
    print(len(data.loc[:,0]))

    #Test for clean data
    cleaned_data = Preprocessor.clean_data(data)
    print(cleaned_data)
    print(cleaned_data[5]) 