#
#
# digits.py
#
#

import numpy as np
from sklearn import cross_validation
import pandas as pd

# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('digits.csv', header=0)
# df.head()
# df.info()

# Convert feature columns as needed...
# You may to define a function, to help out:
def transform(s):
    """ from number to string
    """
    return 'digit ' + str(s)
    
df['label'] = df['64'].map(transform)  # apply the function to the column
# print("+++ End of pandas +++\n")

# import sys
# sys.exit(0)

# print("+++ Start of numpy/scikit-learn +++")

# We'll stick with numpy - here's the conversion to a numpy array
X_data = df.iloc[:,0:64].values        # iloc == "integer locations" of rows/cols
y_data = df[ 'label' ].values      # also addressable by column name(s)

#
# you can divide up your dataset as you see fit here...
#

# X_test = X_data[22:50,0:64]              # the final testing data
# X_train = X_data[50:,0:64]              # the training data

# y_test = y_data[22:50]                  # the final testing outputs/labels (unknown)
# y_train = y_data[50:] 

# going to treat the complete and incomplete data differently, so we need to differentiate it here

X_test_final_incomplete  =X_data[0:10, 0:64]
X_test_final = X_data[10:22,0:64]              # the final testing data
X_train_final = X_data[22:,0:64]              # the training data

y_test_incomplete = y_data[0:10]
y_test_final = y_data[10:22]                  # the final testing outputs/labels (unknown)
y_train_final = y_data[22:]  




#
# feature display - use %matplotlib to make this work smoothly
#
from matplotlib import pyplot as plt

def show_digit( Pixels ):
    """ input Pixels should be an np.array of 64 integers (from 0 to 15) 
        there's no return value, but this should show an image of that 
        digit in an 8x8 pixel square
    """
    # print(Pixels.shape)
    Patch = Pixels.reshape((8,8))
    plt.figure(1, figsize=(4,4))
    plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # cm.gray_r   # cm.hot
    plt.show()
    
# try it!
# row = 6
# Pixels = X_data[row:row+1,:]
# show_digit(Pixels)
# print("That image has the label:", y_data[row])



from sklearn.neighbors import KNeighborsClassifier

print("Predictions for Row 10-22")
print()

knn = KNeighborsClassifier(n_neighbors=1)   # 7 is the "k" in kNN
knn.fit(X_train_final, y_train_final)

print("digit_test's predicted outputs are:")
print()
x_results = knn.predict(X_test_final)

print('Predicted -- Actual')
for x in range(len(x_results)):
    print(x_results[x], ', ', y_test_final[x],'\n')

# and here are the actual labels (iris types)


print()
print("-------------------")
print()



#
# feature engineering...
#


# after predicting full data numbers, transform data to retrain data to predict half info numbers

X_data[:,39:] *= 0   # maybe the first column is worth 100x more!



# bestk = [0,0]

# for k in range(1,6):
#     print(k)

#     knn = KNeighborsClassifier(n_neighbors=k)   # 7 is the "k" in kNN

#     train = []
#     test = []

#     for i in range(0,10):
#         print(i)

#         cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
#             cross_validation.train_test_split(X_train, y_train, test_size=0.25) # random_state=0 

#         # fit the model using the cross-validation data
#         #   typically cross-validation is used to get a sense of how well it works
#         #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        
#         knn.fit(cv_data_train, cv_target_train) 
#         # print("KNN cv training-data score:", knn.score(cv_data_train,cv_target_train))
#         # print("KNN cv testing-data score:", knn.score(cv_data_test,cv_target_test))
#         train.append(knn.score(cv_data_train,cv_target_train))
#         test.append(knn.score(cv_data_test,cv_target_test))

#         avgtrain = sum(train)/10
#         avgtest = sum(test)/10

#         if avgtest > bestk[0]:
#             bestk[0] = avgtest
#             bestk[1] = k

# print(bestk)




print("Predictions for Rows 0-9")
print()

knn = KNeighborsClassifier(n_neighbors=1)   # 7 is the "k" in kNN
knn.fit(X_train_final, y_train_final)

print("digit_test's predicted outputs are:")
x_results_incomplete = knn.predict(X_test_final_incomplete)
print()

print('Predicted -- Actual')
for x in range(len(x_results_incomplete)):
    print(x_results_incomplete[x], ', ', y_test_incomplete[x],'\n')

#
# here, you'll implement the kNN model
#





# knn = KNeighborsClassifier(n_neighbors=1)   # 7 is the "k" in kNN
# knn.fit(X_train, y_train)

# print("digit_test's predicted outputs are:")
# print(knn.predict(X_test))
# print()

# # and here are the actual labels (iris types)
# print("and the actual labels are")
# print(y_test)


#
# run cross-validation
#

# ks_to_consider = []

# num_trials = 10

# for x in range(num_trials):
#     print(x)

#     bestk = [0,0]

#     # for k in range(1,15):

#     knn = KNeighborsClassifier(n_neighbors=1)   # 7 is the "k" in kNN

#     train = []
#     test = []

#     for i in range(0,10):

#         cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
#             cross_validation.train_test_split(X_train_final, y_train_final, test_size=0.25) # random_state=0 

#         # fit the model using the cross-validation data
#         #   typically cross-validation is used to get a sense of how well it works
#         #   and tune any parameters, such as the k in kNN (3? 5? 7? 41?, etc.)
        
#         knn.fit(cv_data_train, cv_target_train) 
#         # print("KNN cv training-data score:", knn.score(cv_data_train,cv_target_train))
#         # print("KNN cv testing-data score:", knn.score(cv_data_test,cv_target_test))
#         train.append(knn.score(cv_data_train,cv_target_train))
#         test.append(knn.score(cv_data_test,cv_target_test))

#         avgtrain = sum(train)/10
#         avgtest = sum(test)/10

#         if avgtest > bestk[0]:
#             bestk[0] = avgtest
#             # bestk[1] = k

#     ks_to_consider.append(bestk[0])

# print(ks_to_consider)
# avg_best_k = sum(ks_to_consider)/num_trials
# print('Best K ==', avg_best_k)



#
# and then see how it does on the two sets of unknown-label data... (!)
#





"""
Comments and results:

Briefly mention how this went:
  + 1
  + very smoothly
  + ~95%

  +had to treat complete and incomplete digits differently!
  +reweighted bottom half data to 0!



Then, include the predicted labels of the 12 digits with full data but no label:
Past those labels (just labels) here:
You'll have 12 lines:

Predicted -- Actual
digit 9 ,  digit -1 

digit 9 ,  digit -1 

digit 5 ,  digit -1 

digit 5 ,  digit -1 

digit 6 ,  digit -1 

digit 5 ,  digit -1 

digit 0 ,  digit -1 

digit 3 ,  digit -1 

digit 8 ,  digit -1 

digit 9 ,  digit -1 

digit 8 ,  digit -1 

digit 4 ,  digit -1 



And, include the predicted labels of the 10 digits that are "partially erased" and have no label:
Mention briefly how you handled this situation!?

I deweighted all of the columns with missing data to make them less important to the model.

Performed cross validation again to check accuracy and find appropriate k value.

Found best k value to again be 1 with accuracy ~ 95%

Past those labels (just labels) here:
You'll have 10 lines:

Predicted -- Actual
digit 0 ,  digit -1 

digit 0 ,  digit -1 

digit 0 ,  digit -1 

digit 1 ,  digit -1 

digit 7 ,  digit -1 

digit 2 ,  digit -1 

digit 3 ,  digit -1 

digit 4 ,  digit -1 

digit 0 ,  digit -1 

digit 1 ,  digit -1


"""