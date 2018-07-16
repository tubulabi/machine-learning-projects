import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

df = pd.read_csv("C:\\Users\\Abinash\\MMT_HackThon\\train.csv")
df_test = pd.read_csv("C:\\Users\\Abinash\\MMT_HackThon\\test.csv")

#Clean train data
number = LabelEncoder()
df['A'] = number.fit_transform(df['A'].astype('str'))
df['A']= df['A'].fillna(0)

df['B'] = df['B'].fillna(df['B'].mean())

df['D'] = number.fit_transform(df['D'].astype('str'))
df['D']= df['D'].fillna(2)

df['E'] = number.fit_transform(df['E'].astype('str'))
df['E']= df['E'].fillna(0)

#df['G'] = number.fit_transform(df['G'].astype('str'))
#df['G']= df['G'].fillna(8)

df['I'] = number.fit_transform(df['I'].astype('str'))
df['J'] = number.fit_transform(df['J'].astype('str'))
df['L'] = number.fit_transform(df['L'].astype('str'))
df['M'] = number.fit_transform(df['M'].astype('str'))
df['N'] = df['N'].fillna(df['N'].mean())


#Clean test data set
df_test['A'] = number.fit_transform(df_test['A'].astype('str'))
df_test['A']= df_test['A'].fillna(0)

df_test['B'] = df_test['B'].fillna(df_test['B'].mean())

df_test['D'] = number.fit_transform(df_test['D'].astype('str'))
df_test['D']= df_test['D'].fillna(2)

df_test['E'] = number.fit_transform(df_test['E'].astype('str'))
df_test['E']= df_test['E'].fillna(0)



df_test['I'] = number.fit_transform(df_test['I'].astype('str'))
df_test['J'] = number.fit_transform(df_test['J'].astype('str'))
df_test['L'] = number.fit_transform(df_test['L'].astype('str'))
df_test['M'] = number.fit_transform(df_test['M'].astype('str'))
df_test['N'] = df_test['N'].fillna(df_test['N'].mean())

# creating testing and training set
X = df[["A","B","C","D","E","H","I","J","K","L","M","N","O"]]
X = np.array(X)


standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

#robust_scaler = RobustScaler()
#X = robust_scaler.fit_transform(X)
#X = min_max_scaler.fit_transform(X)

#target = np.array(df["P"])
Y= np.array(df["P"])

test_data = df_test[["A","B","C","D","E","H","I","J","K","L","M","N","O"]]
test_data = np.array(test_data)
#test_data = robust_scaler.fit_transform(test_data)
#test_data = standard_scaler.fit_transform(test_data)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

# train scikit learn model 
clf = LogisticRegression()
clf.fit(X_train,Y_train)
pred = clf.predict(test_data)
#clf = GaussianNB()
#clf.fit(X_train,Y_train)
#pred = clf.predict(X_test)
pred_on_test_data = clf.predict(test_data)

print(pred)
print(pred_on_test_data)
#prediction = pd.DataFrame(pred, columns=['predictions']).to_csv('MMT_Output1.csv')



#print('Score Scikit learn:', clf.score(X,target))
#print(confusion_matrix(pred,target[:138]))
#print(classification_report(pred,target[:138]))
#print(hamming_loss(pred,target[:138]))

print('Score Scikit learn:', clf.score(X_train,Y_train))
print('Confusion Matrix:', confusion_matrix(Y_test, pred))
print('Classification Report:', classification_report(Y_test, pred))
print('Hamming Lose:', hamming_loss(Y_test, pred))


#Plot the ROC curve
logit_roc_auc = roc_auc_score(Y_test,pred)
fpr, tpr, thresholds = roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
