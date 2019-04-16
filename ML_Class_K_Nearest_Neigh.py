# <a href="https://www.bigdatauniversity.com">
# <img src = "https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width = 400, align = "center"></a>
# 
# In this lab exercise, you will learn a popular machine learning algorithm, Decision Tree. 
# You will use this classification algorithm to build a model from historical data of patients, 
# and their respond to different medications. 
# Then you use the trained decision tree to predict the class of a unknown patient, 
# or to find a proper drug for a new patient.
#
# Import the Following Libraries:
# <ul>
#     <li> <b>numpy (as np)</b> </li>
#     <li> <b>pandas</b> </li>
#     <li> <b>DecisionTreeClassifier</b> from <b>sklearn.tree</b> </li>
# </ul>

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import wget

# ### About dataset
# Imagine that you are a medical researcher compiling data for a study. 
# You have collected data about a set of patients, all of whom suffered from the same illness. 
# During their course of treatment, each patient responded to one of 5 medications,
# Drug A, Drug B, Drug c, Drug x and y. 
# 
# Part of your job is to build a model to find out which drug might be appropriate 
# for a future patient with the same illness. 
# The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, 
# and the target is the drug that each patient responded to. 
# 
# It is a sample of binary classifier, and you can use the training part of the dataset 
# to build a decision tree, and then use it to predict the class of a unknown patient, 
# or to prescribe it to a new patient.
# 

# ### Downloading Data
# To download the data, we will use !wget to download it from IBM Object Storage.

# get_ipython().system('wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv')

wget.download('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv')

# now, read data using pandas dataframe:

my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]

# ## Pre-processing
# Using <b>my_data</b> as the Drug.csv data read by pandas, declare the following variables: <br>
# <ul>
#     <li> <b> X </b> as the <b> Feature Matrix </b> (data of my_data) </li>
# 
#     
#     <li> <b> y </b> as the <b> response vector (target) </b> </li>
# 
# 
#    
# </ul>
# Remove the column containing the target name since it doesn't contain numeric values.

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

# As you may figure out, some featurs in this dataset are catergorical such as __Sex__ or __BP__.
# Unfortunately, Sklearn Decision Trees do not handle categorical variables. 
# But still we can convert these features to numerical values. 
# __pandas.get_dummies()__
# Convert categorical variable into dummy/indicator variables.

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

# Now we can fill the target variable.

y = my_data["Drug"]
y[0:5]

# ## Setting up the Decision Tree
# We will be using <b>train/test split</b> on our <b>decision tree</b>. 
# Let's import <b>train_test_split</b> from <b>sklearn.cross_validation</b>.

from sklearn.model_selection import train_test_split

# Now <b> train_test_split </b> will return 4 different parameters. 
# We will name them:<br>
# X_trainset, X_testset, y_trainset, y_testset <br> <br>
# The <b> train_test_split </b> will need the parameters: <br>
# X, y, test_size=0.3, and random_state=3. <br> <br>
# The <b>X</b> and <b>y</b> are the arrays required before the split, 
# the <b>test_size</b> represents the ratio of the testing dataset, 
# and the <b>random_state</b> ensures that we obtain the same splits.

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# ## Practice
# Print the shape of X_trainset and y_trainset. Ensure that the dimensions match

# Print the shape of X_testset and y_testset. Ensure that the dimensions match

# ## Modeling
# We will first create an instance of the <b>DecisionTreeClassifier</b> called <b>drugTree</b>.<br>
# Inside of the classifier, specify <i> criterion="entropy" </i> so we can see the information gain of each node.

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

# Next, we will fit the data with the training feature matrix <b> X_trainset </b> and training  response vector <b> y_trainset </b>

drugTree.fit(X_trainset,y_trainset)

# ## Prediction
# Let's make some <b>predictions</b> on the testing dataset and store it into a variable called <b>predTree</b>.

predTree = drugTree.predict(X_testset)

# You can print out <b>predTree</b> and <b>y_testset</b> if you want to visually compare the prediction to the actual values.

print ("")
print (predTree [0:5])
print (y_testset [0:5])

# ## Evaluation
# Next, let's import __metrics__ from sklearn and check the accuracy of our model.

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# __Accuracy classification score__ computes subset accuracy: 
# the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.  
# 
# In multilabel classification, the function returns the subset accuracy. 
# If the entire set of predicted labels for a sample strictly match with the true set of labels, 
# then the subset accuracy is 1.0; otherwise it is 0.0.