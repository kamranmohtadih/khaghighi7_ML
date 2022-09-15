import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm


# Read the data************************************************
df1 = pd.read_csv('data2.csv', index_col=0, header=0)
df1.fillna(0)
df1.dropna()
x = df1.iloc[:,:-1]
y = df1.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
encoder = preprocessing.LabelEncoder()
df1 = df1.applymap(str)
df1 = df1.apply(lambda x: x.str.strip())
for i in range(df1.columns.size-1):
    x_train.iloc[:,i] = encoder.fit_transform(x_train.iloc[:,i])
    x_test.iloc[:, i] = encoder.fit_transform(x_test.iloc[:, i])



# #Deceision tree *********************************************
# *************************************************************
# *************************************************************


# Max depth dependency graphs**********************************
max_depth_list = range(1,10)
train_errors = []
test_errors = []
for i in max_depth_list:
    dtc = DecisionTreeClassifier(max_depth =i, criterion="gini")
    dtc.fit(x_train,y_train)
    pred_train = dtc.predict(x_train)
    pred_test = dtc.predict(x_test)
    train_errors.append(metrics.accuracy_score(y_train, pred_train))
    test_errors.append(metrics.accuracy_score(y_test, pred_test))
j = range(1,10)
plt.plot(j, train_errors, label='Training Accuracy')
plt.plot(j, test_errors, label='Testing Accuracy')
plt.xlabel('Max Depth') # Label x-axis
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('images/DT_Accuracy_max depth.png')
plt.clf()
plt.close('images/DT_Accuracy_max depth.png')

# Learning curves ***********************************
#From previous section we understood that max_depth = 4 is the best for both datasets
clf = DecisionTreeClassifier(max_depth =4, criterion="entropy" )
clf.fit(x_train, y_train)
for i in range(df1.columns.size-1):
    x.iloc[:,i] = encoder.fit_transform(x.iloc[:,i])
    y = encoder.fit_transform(y)
print(x.shape)
print(y.shape)
train_sizes, train_scores, test_scores = learning_curve(clf, x,y,  n_jobs=-1, train_sizes=np.linspace(0.001, 1, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.subplots(1, figsize=(10, 10))
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#234561", label="Test score")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('images/DT_Learning_curve.png')
plt.clf()
plt.close('images/DT_Learning_curve.png')

# Post pruning **********************************
path = clf.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas,impurities)
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=34, ccp_alpha=ccp_alpha)
    clf.fit(x_train,y_train)
    clfs.append(clf)

train_acc = []
test_acc =[]
for c in clfs:
    y_train_pred = c.predict(x_train)
    y_test_pred = c.predict(x_test)
    train_acc.append(metrics.accuracy_score(y_train_pred,y_train))
    test_acc.append(metrics.accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc,color="#111111")
plt.scatter(ccp_alphas,test_acc,color="#7CFC00")
plt.plot(ccp_alphas,train_acc,label="train_accuracy")
plt.plot(ccp_alphas,test_acc,label="test_accuracy")
plt.legend()
plt.title("Decision tree with pruning")
plt.savefig('images/DT_pruning.png')
plt.clf()
plt.close('images/DT_pruning.png')
# #Neural networks - MLP *********************************************
# *************************************************************
# *************************************************************
clf = MLPClassifier(solver='sgd', hidden_layer_sizes =(3,5), random_state=34, activation='relu')
clf.fit(x_train, y_train)
for i in range(df1.columns.size-1):
    x.iloc[:,i] = encoder.fit_transform(x.iloc[:,i])
    y = encoder.fit_transform(y)

print(x.shape)
print(y.shape)
train_sizes, train_scores, test_scores = learning_curve(clf, x,y,  n_jobs=-1, train_sizes=np.linspace(0.001, 1, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.subplots(1, figsize=(10, 10))
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#234561", label="Test score")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('images/MLP_Learning_curve.png')
plt.clf()
plt.close('images/MLP_Learning_curve.png')
# #Boosting *********************************************
# *************************************************************
# *************************************************************
# Max depth dependency graphs**********************************
max_depth_list = range(1,10)
train_errors = []
test_errors = []
for i in max_depth_list:
    dtc = GradientBoostingClassifier(max_depth =i, )
    dtc.fit(x_train,y_train)
    pred_train = dtc.predict(x_train)
    pred_test = dtc.predict(x_test)
    train_errors.append(metrics.accuracy_score(y_train, pred_train))
    test_errors.append(metrics.accuracy_score(y_test, pred_test))
j = range(1,10)
plt.plot(j, train_errors, label='Training Accuracy')
plt.plot(j, test_errors, label='Testing Accuracy')
plt.xlabel('Max Depth') # Label x-axis
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('images/Boosted_Accuracy_max depth.png')
plt.clf()
plt.close('images/Boosted_Accuracy_max depth.png')

# Learning curves ***********************************
#From previous section we understood that max_depth = 4 is the best for both datasets
clf = GradientBoostingClassifier(max_depth =4 )
clf.fit(x_train, y_train)
for i in range(df1.columns.size-1):
    x.iloc[:,i] = encoder.fit_transform(x.iloc[:,i])
    y = encoder.fit_transform(y)
print(x.shape)
print(y.shape)
train_sizes, train_scores, test_scores = learning_curve(clf, x,y,  n_jobs=-1, train_sizes=np.linspace(0.001, 1, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.subplots(1, figsize=(10, 10))
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#234561", label="Test score")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('images/Boosting_Learning_curve.png')
plt.clf()
plt.close('images/Boosting_Learning_curve.png')

# Post pruning for boosting**********************************
clf = DecisionTreeClassifier(max_depth =4, criterion="entropy" )
path = clf.cost_complexity_pruning_path(x_train,y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas,impurities)
clfs = []
for ccp_alpha in ccp_alphas:
    clf = GradientBoostingClassifier(random_state=34, ccp_alpha=ccp_alpha)
    clf.fit(x_train,y_train)
    clfs.append(clf)

train_acc = []
test_acc =[]
for c in clfs:
    y_train_pred = c.predict(x_train)
    y_test_pred = c.predict(x_test)
    train_acc.append(metrics.accuracy_score(y_train_pred,y_train))
    test_acc.append(metrics.accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc,color="#111111")
plt.scatter(ccp_alphas,test_acc,color="#7CFC00")
plt.plot(ccp_alphas,train_acc,label="train_accuracy")
plt.plot(ccp_alphas,test_acc,label="test_accuracy")
plt.legend()
plt.title("Boosted Decision tree with pruning")
plt.savefig('images/Boosting_pruning.png')
plt.clf()
plt.close('images/Boosting_pruning.png')

# #Boosting *********************************************
# *************************************************************
# *************************************************************
clf = svm.SVC(kernel='poly')
clf.fit(x_train, y_train)
for i in range(df1.columns.size-1):
    x.iloc[:,i] = encoder.fit_transform(x.iloc[:,i])
    y = encoder.fit_transform(y)
print(x.shape)
print(y.shape)
train_sizes, train_scores, test_scores = learning_curve(clf, x,y,  n_jobs=-1, train_sizes=np.linspace(0.001, 1, 100))
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.subplots(1, figsize=(10, 10))
plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
plt.plot(train_sizes, test_mean, color="#234561", label="Test score")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('images/SVM_Learning_curve.png')
plt.clf()
plt.close('images/SVM_Learning_curve.png')