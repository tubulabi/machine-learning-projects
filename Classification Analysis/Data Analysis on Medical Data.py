

##To find the unique levels

#Question1
import pandas as pd
import numpy as np
df = pd.read_csv("C:\\Users\\annotation.csv")

#Count of Type in each coloums
df1 = df.groupby('Dr. 123 Kid')['cell_id'].nunique()
df2 = df.groupby('Dr. Do Little')['cell_id'].nunique()
df3 = df.groupby('Dr. Scrubs')['cell_id'].nunique()
print(df1.head())
print(df2.head())
print(df3.head())


#Compare three coloums to find out common Type among three doctors
#def compare_cols(row_input):
 #   if row_input['Dr. 123 Kid'] == row_input['Dr. Do Little'] and row_input['Dr. Do Little'] == row_input['Dr. Scrubs'] and row_input['Dr. 123 Kid'] == row_input['Dr. Scrubs']:
  #      return True
   # return False

#df.apply(compare_cols, axis = 1)

df['result'] = (df['Dr. 123 Kid'] == df['Dr. Do Little']) & (df['Dr. Do Little'] == df ['Dr. Scrubs'])
print((df.head))
print('Number of cells aggrements in label from the 3 experts :', (df['result']==True).sum())
print('Number of cells disaggrements in label from the 3 experts :', (df['result']==False).sum())



#Question 2: All labels are different
df['three_label_not_equal'] = (df['Dr. 123 Kid'] != df['Dr. Do Little']) & (df['Dr. Do Little'] != df ['Dr. Scrubs']) 
print((df.head()))
print('Number of cells where no labels are equal :', (df['three_label_not_equal']==True).sum())
print("Type5 Identified as most common type among all doctors")


#Question 3: Calculate Precision, recall and confidence inteval for labels that has only two labels are equal
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import hamming_loss
from sklearn.model_selection import cross_val_score


df['two_label'] = (df['Dr. 123 Kid'] == df['Dr. Do Little']) | (df['Dr. Do Little'] == df ['Dr. Scrubs']) | (df['Dr. 123 Kid'] == df['Dr. Scrubs'])
print((df.head()))
print('Number of cells aggrements in label from at least 2 experts :', (df['two_label']==True).sum())
print('Number of cells disaggrements in label from at least 2 experts :', (df['two_label']==False).sum())

#Transfer data set
#Remove Integer from String
df['Dr. 123 Kid'] = df['Dr. 123 Kid'].str.extract('(\d+)')
df['Dr. Do Little'] = df['Dr. Do Little'].str.extract('(\d+)')
df['Dr. Scrubs'] = df['Dr. Scrubs'].str.extract('(\d+)')

print((df.head()))

# creating testing and training set
X = df[["Dr. 123 Kid","Dr. Do Little", "Dr. Scrubs"]]
X = np.array(X)

Y= np.array(df["two_label"])
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

# train scikit learn model 
#clf = LogisticRegression()
clf =tree.DecisionTreeClassifier()
clf.fit(X_train,Y_train)
pred = clf.predict(X_test)

#pred_on_test_data = clf.predict(test_data)

#K-Fold cross validation
accuracies = cross_val_score(estimator = clf, X= X_train, y = Y_train, cv=10)
print('Total accuracy after cross validation:', accuracies.mean())
print('Standard deviation of the model after cross validation :',accuracies.std())

print('Score Scikit learn:', clf.score(X_train,Y_train))
print('Confusion Matrix:', confusion_matrix(Y_test, pred))
print('Classification Report:', classification_report(Y_test, pred))
print('Hamming Lose:', hamming_loss(Y_test, pred))


#Question 4:: Rating of the each cell based on doctors annotations

df = pd.read_csv("C:\\Users\\annotation.csv")

#Any two same
df['two_label_equal'] = (df['Dr. 123 Kid'] == df['Dr. Do Little']) | (df['Dr. Do Little'] == df ['Dr. Scrubs']) | (df['Dr. 123 Kid'] == df['Dr. Scrubs'])

#All three different
df['three_label_not_equal'] = (df['Dr. 123 Kid'] != df['Dr. Do Little']) & (df['Dr. Do Little'] != df ['Dr. Scrubs'])

#All are same
df['all_equal'] = (df['Dr. 123 Kid'] == df['Dr. Do Little']) & (df['Dr. Do Little'] == df ['Dr. Scrubs'])

conditions = [
    (df['all_equal'] == True),
    (df['two_label_equal'] == True),
    (df['three_label_not_equal'] == True)]
rating_num = [5, 3, 1]
df['rating'] = np.select(conditions, rating_num)
print(df.head(10))
    