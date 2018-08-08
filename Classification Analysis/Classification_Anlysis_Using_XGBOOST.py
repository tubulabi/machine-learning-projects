#Import libraries:
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV   
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import StratifiedKFold 


import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv("...\\Prob_Stat_2\\train.csv", header=0)
test = pd.read_csv("...\\Prob_Stat_2\\test.csv", header=0)



#print(train.head())
#print(test.head())


X = train.iloc[:,1:55]
X = np.array(X)

X_test_to_pred = test.iloc[:,1:55]
X_test_to_pred = np.array(X_test_to_pred)

#big_X = train + test

#X_train = big_X[0:train.shape[0]].as_matrix()
#X_test = big_X[train.shape[0]::].as_matrix()
#Y_train = train['label']
#from sklearn.preprocessing import StandardScaler
#standard_scaler = StandardScaler()
#X = standard_scaler.fit_transform(X)
#Y= np.array(train["label"])
Y = train['label'].values

seed = 500
np.random.seed(seed)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state = seed)

params = {
        
        'max_depth':[3, 4, 5,8],
        'min_child_weight':[1, 5, 10]
        }
#param_test1 = {
# 'max_depth':np.arange(3,10,2),
# 'n_estimators':[5,10,25,50],
 # 'learning_rate':np.linspace(0.001, 1, 3)  
#}

param_fixed = {
    'objective':'binary:logistic',
    'silent':1,
    'max_depth':5,
    'min_child_weight':1, 
    'gamma':0, 
    'subsample':0.8, 
    'colsample_bytree':0.8
}

#bst = XGBClassifier(**params).fit(X_train, Y_train))
#bst_grid = GridSearchCV(estimator = XGBClassifier(**params),param_grid = param_test1)
#train_predictions = bst.predict(X_test)
#train_predprob = bst.predict_proba(X_test)

 #scale_pos_weight=1, seed=27, colsample_bytree=0.8,
#skf = StratifiedKFold(Y,n_folds= 10, shuffle = True, random_state = seed)

gsearch1 = GridSearchCV(estimator = XGBClassifier(**param_fixed, learning_rate=0.02, n_estimators=600, seed =seed), 
 param_grid = params, scoring='roc_auc', iid=False, cv=5, verbose = 5) #n_jobs=4

bst = gsearch1.fit(X_train,Y_train)
bst.grid_scores_, bst.best_params_, bst.best_score_


train_predictions = bst.predict(X_test)
train_predprob = bst.predict_proba(X_test)
dtest_predprob = bst.predict_proba(X_test_to_pred)

print(dtest_predprob)

Check the accuracy of the model
correct = 0
for i in range(len(train_predictions)):
    if(Y_test[i] == train_predictions[i]):
        correct +=1
        
acc = accuracy_score(Y_test,train_predictions)

print("\nmodel report:")
print('Predicted correctly: {0}/{1}'.format(correct, len(train_predictions)))
print("Accuracy is:", acc)
print('Error: {0:.4f}'.format(1-acc))









