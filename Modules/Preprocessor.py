'''
Authors: Fasil Cheema and Mahmoud Alsaeed
Purpose: This module preprocesses the data (tweets) in preperation to create a network
'''

import os 

class Preprocessor():

    def initialize_dir(self):
        path_parent = os.path.dirname(os.getcwd())
        os.chdir(path_parent)

if __name__ == '__main__':
    Preprocessor = Preprocessor()
    print(Preprocessor.initialize())
    print('end')