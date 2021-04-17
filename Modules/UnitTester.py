from GraphGenerator import GraphGenerator
from Preprocessor import Preprocessor

#%% Create instances of each class to be tested
GraphGenerator = GraphGenerator()
Preprocessor = Preprocessor()


#%% Test Case for Preprocessor 
# initialize class
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

#%% Test Case for GraphGenerator
#tweets for test cases
tweet1 = "@Mention"
tweet2 = "Keyword"
tweet3 = "#Hashtag"
tweet4 = "@Mention #Hashtag Keyword"
tweet5 = "@Mention1 Keyword1 @Mention2 Keyword2"
tweet6 = "Keyword1 Keyword2 #Hashtag1 @Mention1 Keyword3 Keyword4 @Mention2 #Hashtag2 Keyword5"

print("starting test")

Ah_1, Ak_1, Am_1, Bhk_1, Bhm_1, Bkh_1, Bkm_1, Bmh_1, Bmk_1 = GraphGenerator.MatrixGenerator(tweet1)

print('Test 1')
print(Ah_1, Ak_1, Am_1, Bhk_1, Bhm_1, Bkh_1, Bkm_1, Bmh_1, Bmk_1)

Ah_2, Ak_2, Am_2, Bhk_2, Bhm_2, Bkh_2, Bkm_2, Bmh_2, Bmk_2 = GraphGenerator.MatrixGenerator(tweet2)

print('Test 2')
print(Ah_2, Ak_2, Am_2, Bhk_2, Bhm_2, Bkh_2, Bkm_2, Bmh_2, Bmk_2)

Ah_3, Ak_3, Am_3, Bhk_3, Bhm_3, Bkh_3, Bkm_3, Bmh_3, Bmk_3 = GraphGenerator.MatrixGenerator(tweet3)

print('Test 3')
print(Ah_3, Ak_3, Am_3, Bhk_3, Bhm_3, Bkh_3, Bkm_3, Bmh_3, Bmk_3)

Ah_4, Ak_4, Am_4, Bhk_4, Bhm_4, Bkh_4, Bkm_4, Bmh_4, Bmk_4 = GraphGenerator.MatrixGenerator(tweet4)

print('Test 4')
print(Ah_4, Ak_4, Am_4, Bhk_4, Bhm_4, Bkh_4, Bkm_4, Bmh_4, Bmk_4)

Ah_5, Ak_5, Am_5, Bhk_5, Bhm_5, Bkh_5, Bkm_5, Bmh_5, Bmk_5 = GraphGenerator.MatrixGenerator(tweet5)

print('Test 5')
print(Ah_5, Ak_5, Am_5, Bhk_5, Bhm_5, Bkh_5, Bkm_5, Bmh_5, Bmk_5)

Ah_6, Ak_6, Am_6, Bhk_6, Bhm_6, Bkh_6, Bkm_6, Bmh_6, Bmk_6 = GraphGenerator.MatrixGenerator(tweet6)

print('Test 6')
print(Ah_6)
print('next')
print(Ak_6) 
print(' next ')
print(Am_6)
print(' next ')
print(Bhk_6)
print(' next ')
print(Bhm_6)
print(' next ')
print(Bkh_6)
print(' next ')
print(Bkm_6)
print(' next ')
print(Bmh_6) 
print(' next ')
print(Bmk_6)

