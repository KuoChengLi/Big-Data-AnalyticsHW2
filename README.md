# Big-Data-AnalyticsHW2
## *1.  Parameter Tuning in XGBoost*

### importing the required libraries and loading the data:
```
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV   
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 6

data = pd.read_csv('C:/Users/MAX/Desktop/LargeTrain.csv')
train = pd.DataFrame(data)
a= lambda x: x-1                          
train['Class']=train['Class'].apply(a)     #change the range of column Class
```
### initial estimating
```
target='Class'
predictors = [x for x in train.columns if x not in [target]]
xgb0 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb0, train, predictors)
```
*  Accuracy : 0.9967
### tuning max_depth and min_child_weight
```
param_test1 = {
 'max_depth':range(3,10),
 'min_child_weight':range(1,6)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
*  ideal values : <strong>max_depth: 9</strong><strong>, min_child_weight: 1</strong>
### tuning gamma
```
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```
*  ideal value : <strong>gamma: 0.3</strong>
### tuning subsample
```
param_test3 = {
'subsample':[i/100.0 for i in range(60,100)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1, max_depth=9,
 min_child_weight=1, gamma=0.3, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3,n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```
*  ideal value : <strong>subsample: 0.92</strong>

### tuning colsample_bytree
```
param_test4 = {
 'colsample_bytree':[i/100.0 for i in range(60,100)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1, max_depth=9,
 min_child_weight=1, gamma=0.3, subsample=0.92, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test4,n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```
*  ideal value : <strong>colsample_bytree: 0.93</strong>
### tuning reg_alpha
```
param_test5 = {
 'reg_alpha':[1e-5, 1e-2,0.1,1,100]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1, max_depth=9,
 min_child_weight=1, gamma=0.3, subsample=0.92, colsample_bytree=0.93,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test5,n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

param_test6 = {
 'reg_alpha':[0,0.01,0.05,0.1,0.5]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1, max_depth=9,
 min_child_weight=1, gamma=0.3, subsample=0.92, colsample_bytree=0.93,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test6,n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
```
*  ideal value : <strong>reg_alpha: 0.05</strong>

### estimating by using the updated parameters
*  max_depth=9
*  min_child_weight=1
*  gamma=0.3
*  subsample=0.92
*  colsample_bytree =0.93
*  reg_alpha=0.05
```
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=1,
 gamma=0.3,
 subsample=0.92,
 colsample_bytree=0.93,
 reg_alpha=0.05,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)
```
*  Accuracy : 0.999
### using confusion matrix to evaluate the quality
```
def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):
    plt.imshow(cm , interpolation='nearest' , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks , classes , rotation=45)
    plt.yticks(tick_marks , classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    else:
        print('Confusion matrix , without normalization')
    #print(cm)
    
    thresh = cm.max()/2.
    for i , j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
        plt.text(j , i , cm[i,j] , horizontalalignment='center' , color='white' if cm[i,j]>thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    
class_name = ['Class' + str(x) for x in range(1,10)]
target='Class'
predictors = [x for x in train.columns if x not in [target]]
X = data[predictors]
y = data[target]

X_train , X_test , y_train , y_test = train_test_split(X, y , random_state=0)
clf = XGBClassifier(max_depth=9,min_child_weight=1,gamma=0.3,subsample=0.92,colsample_bytree=0.93,reg_alpha=0.05)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test , y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_name,title='Confusion matrix')
plt.show
```
![Image of Confusion matrix](http://imgur.com/ezFXert.jpg)
## *2.  Tuning in Gradient Boosting*
### importing the required libraries and loading the data:
```
import itertools
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 6
```
### tuning n_estimators
```
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1,n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```
*  ideal value : <strong>n_estimators: 80</strong>
### tuning max_depth and min_samples_split
```
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, 
max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
```
*  ideal values : <strong>max_depth: 9</strong><strong>, min_samples_split: 200</strong>
### tuning min_samples_leaf
```
param_test3 = {'min_samples_leaf':range(10,71,10)}
grid3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,
max_depth =9,min_samples_split=200,max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test3,n_jobs=4,iid=False, cv=5)
grid3.fit(train[predictors],train[target])
grid3.grid_scores_, grid3.best_params_, grid3.best_score_
```
*  ideal value : <strong>min_samples_leaf: 20</strong><strong>
### tuning max_features
```
param_test4 = {'max_features':range(7,50,2)}
grid4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
n_estimators=80,max_depth =9,min_samples_split=200, subsample=0.8, random_state=10,min_samples_leaf=20)
,param_grid = param_test4,n_jobs=4,iid=False, cv=5 )
grid4.fit(train[predictors],train[target])
grid4.grid_scores_, grid4.best_params_, grid4.best_score_
```
*  ideal value : <strong>max_features: 29</strong><strong>
### tuning subsample
```
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5= GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
n_estimators=80,max_depth =9,min_samples_split=200, subsample=0.8, random_state=10,min_samples_leaf=20,max_features=29),
param_grid= param_test5,n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
```
*  ideal value : <strong>subsample: 0.8</strong><strong>
### estimating by using the updated parameters
*  n_estimators: 80
*  max_depth: 9
*  min_samples_split: 200
*  min_samples_leaf: 20
*  max_features: 29
*  subsample: 0.8
```
gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth=9,min_samples_split=200,min_samples_leaf=20,
subsample=0.8,max_features=29,warm_start=True)
modelfit(gbm1, train, predictors)
```
*  Accuracy : 1
*  CV Score : Mean - 0.9954921 | Std - 0.000935757 | Min - 0.9944802 | Max - 0.997235
### using confusion matrix to evaluate the quality
```
def plot_confusion_matrix(cm , classes , normalize=False , title='Confusion matrix' , cmap=plt.cm.Blues):
    plt.imshow(cm , interpolation='nearest' , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks , classes , rotation=45)
    plt.yticks(tick_marks , classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    #else:
        #print('Confusion matrix , without normalization')
    #print(cm)
    
    thresh = cm.max()/2.
    for i , j in itertools.product(range(cm.shape[0]) , range(cm.shape[1])):
        plt.text(j , i , cm[i,j] , horizontalalignment='center' , color='white' if cm[i,j]>thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    
target='Class'
predictors = [x for x in train.columns if x not in [target]]
class_name = ['Class' + str(x) for x in range(1,10)]
X = data[predictors]
y = data[target]

X_train , X_test , y_train , y_test = train_test_split(X, y , random_state=0)
clf = GradientBoostingClassifier(n_estimators=80,max_depth=9,min_samples_split=200,min_samples_leaf=20,subsample=0.8,max_features=29)
clf.fit(X_train , y_train)
y_pred = clf.predict(X_test)

cnf_matrix = confusion_matrix(y_test , y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_name,title='Confusion matrix')
plt.show
```
![Image of Confusion matrix](http://imgur.com/yur6hsa.jpg)
