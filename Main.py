import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from mlxtend.plotting import plot_learning_curves
from plot_learning_curve import plot_learning_curve
from sklearn.model_selection import ShuffleSplit

import time
import warnings
warnings.filterwarnings("ignore")

# Read the data************************************************
df1 = pd.read_csv('data2.csv', index_col=0, header=0)
df1.replace([np.inf, -np.inf], np.nan, inplace=True)
df1.dropna(inplace=True)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
names = df1.columns.values
indexes = df1.index.values
x = df1.iloc[:,:-1]
y = df1.iloc[:,-1]
x = preprocessing.OneHotEncoder().fit_transform(x).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
cv = ShuffleSplit(n_splits=3
                  , test_size=0.2, random_state=0)

# #Deceision tree *********************************************
# *************************************************************
# *************************************************************


# Max depth dependency graphs**********************************
#
max_depth_list = range(1,20)
train_errors = []
test_errors = []

train_errors1 = []
test_errors1 = []
for i in max_depth_list:
    dtc = DecisionTreeClassifier(max_depth =i, criterion="gini" )
    dtc1 = DecisionTreeClassifier(max_depth=i, criterion="entropy")
    dtc.fit(x_train,y_train)
    dtc1.fit(x_train, y_train)

    pred_train = dtc.predict(x_train)
    pred_test = dtc.predict(x_test)

    pred_train1 = dtc1.predict(x_train)
    pred_test1 = dtc1.predict(x_test)
    train_errors.append(metrics.accuracy_score(y_train, pred_train))
    test_errors.append(metrics.accuracy_score(y_test, pred_test))
    train_errors1.append(metrics.accuracy_score(y_train, pred_train1))
    test_errors1.append(metrics.accuracy_score(y_test, pred_test1))
j = range(1,20)
plt.plot(j, train_errors, label='Training Accuracy - Gini')
plt.plot(j, test_errors, label='Testing Accuracy - Gini')
plt.plot(j, train_errors1, label='Training Accuracy - Entropy')
plt.plot(j, test_errors1, label='Testing Accuracy - Entropy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('images/DT_Accuracy_max depth.png')
plt.clf()
plt.close('images/DT_Accuracy_max depth.png')

# Min leaf dependency graph ********************************************
min_leaf_list = range(1,20)
train_errors = []
test_errors = []

train_errors1 = []
test_errors1 = []

for i in min_leaf_list:
    dtc = DecisionTreeClassifier(min_samples_leaf =i, criterion="gini" )
    model = dtc1 = DecisionTreeClassifier(min_samples_leaf=i, criterion="entropy")
    dtc.fit(x_train,y_train)
    dtc1.fit(x_train, y_train)
    pred_train = dtc.predict(x_train)
    pred_test = dtc.predict(x_test)

    pred_train1 = dtc1.predict(x_train)
    pred_test1 = dtc1.predict(x_test)
    train_errors.append(metrics.accuracy_score(y_train, pred_train))
    test_errors.append(metrics.accuracy_score(y_test, pred_test))
    train_errors1.append(metrics.accuracy_score(y_train, pred_train1))
    test_errors1.append(metrics.accuracy_score(y_test, pred_test1))

j = range(1,20)
plt.plot(j, train_errors, label='Training Accuracy - Gini')
plt.plot(j, test_errors, label='Testing Accuracy - Gini')
plt.plot(j, train_errors1, label='Training Accuracy - Entropy')
plt.plot(j, test_errors1, label='Testing Accuracy - Entropy')

plt.xlabel('Min Leaf Sample')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('images/DT_Accuracy_min leaf.png')
plt.clf()
plt.close('images/DT_Accuracy_min leaf.png')

# Learning curves ***********************************

clf = DecisionTreeClassifier(max_depth =6, criterion="entropy" )
plot_learning_curves(x_train,y_train,x_test,y_test,clf,scoring="accuracy")
plt.savefig('images/DT_Learning_curve_entropy_withoutcv.png')
plt.clf()
plt.close('images/DT_Learning_curve_entropy_withoutcv.png')
plot_learning_curve(
   clf,
    "Learning curve DT with entropy",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)


plt.savefig('images/DT_Learning_curve_entropy.png')
plt.clf()
plt.close('images/DT_Learning_curve_entropy.png')


t0 = time.time()
clf = DecisionTreeClassifier(max_depth =6, criterion="gini" )
plot_learning_curve(
   clf,
    "Learning curve DT with entropy",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
print("DT learning curve : ", time.time()-t0)
plt.savefig('images/DT_Learning_curve_gini.png')
plt.clf()
plt.close('images/DT_Learning_curve_gini.png')
plot_learning_curves(x_train,y_train,x_test,y_test,clf,scoring="accuracy")
plt.savefig('images/DT_Learning_curve_gini_withoutcv.png')
plt.clf()
plt.close('images/DT_Learning_curve_gini_withoutcv.png')

# Post pruning **********************************
path = clf.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(max_depth=6,criterion='gini', random_state=34, ccp_alpha=ccp_alpha)
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
plt.title("Decision tree with pruning - Max depth 6 and with gini")
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.savefig('images/DT_pruning.png')
plt.clf()
plt.close('images/DT_pruning.png')

# #Neural networks - MLP *********************************************
# *************************************************************
# *************************************************************
t0 = time.time()
clf = MLPClassifier(solver='sgd', hidden_layer_sizes =(3,5), random_state=34, activation='relu')

plot_learning_curve(
   clf,
    "Learning curve NLP - hidden layer (3,5) and activation function relu",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
print("MLP learning curve : ", time.time()-t0)
plt.savefig('images/MLP_Learning_curve_35_relu.png')
plt.clf()
plt.close('images/MLP_Learning_curve_35_relu.png')
plot_learning_curves(x_train,y_train,x_test,y_test,clf,scoring="accuracy")
plt.savefig('images/MLP_Learning_curve_35_relu_withoutcv.png')
plt.clf()
plt.close('images/MLP_Learning_curve_35_relu_withoutcv.png')

clf = MLPClassifier(solver='sgd', hidden_layer_sizes =(3,5,3), random_state=34, activation='relu')
plot_learning_curve(
   clf,
    "Learning curve NLP - hidden layer (3,5,3) and activation function relu",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
plt.savefig('images/MLP_Learning_curve_353_relu.png')
plt.clf()
plt.close('images/MLP_Learning_curve_353_relu.png')

clf = MLPClassifier(solver='sgd', hidden_layer_sizes =(3,5), random_state=34, activation='identity')
plot_learning_curve(
   clf,
    "Learning curve NLP - hidden layer (3,5) and activation function identity",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
plt.savefig('images/MLP_Learning_curve_35_identity.png')
plt.clf()
plt.close('images/MLP_Learning_curve_35_identity.png')
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
t0 = time.time()
clf = GradientBoostingClassifier(max_depth =6 , criterion='squared_error')
plot_learning_curve(
   clf,
    "Learning curve Gradient booster - criterion squared error",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
print("Gradient learning curve : ", time.time()-t0)
plt.savefig('images/Boosting_Learning_curve.png')
plt.clf()
plt.close('images/Boosting_Learning_curve.png')
plot_learning_curves(x_train,y_train,x_test,y_test,clf,scoring="accuracy")
plt.savefig('images/Boosting_Learning_curve_withoutcv.png')
plt.clf()
plt.close('images/Boosting_Learning_curve_withoutcv.png')

# Post pruning for boosting**********************************
clf = DecisionTreeClassifier(max_depth =6, criterion="gini" )
path = clf.cost_complexity_pruning_path(x_train,y_train)

ccp_alphas, impurities = path.ccp_alphas, path.impurities

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

plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.scatter(ccp_alphas,train_acc,color="#111111")
plt.scatter(ccp_alphas,test_acc,color="#7CFC00")
plt.plot(ccp_alphas,train_acc,label="train_accuracy")
plt.plot(ccp_alphas,test_acc,label="test_accuracy")
plt.legend()
plt.title("Boosted Decision tree with pruning")
plt.savefig('images/Boosting_pruning.png')
plt.clf()
plt.close('images/Boosting_pruning.png')

# #Support vector machine**************************************
# *************************************************************
# *************************************************************
clf = svm.SVC(kernel='poly')
plot_learning_curve(
   clf,
    "SVM learning curve- kernel = poly",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
plt.savefig('images/SVM_Learning_curve_poly.png')
plt.clf()
plt.close('images/SVM_Learning_curve_poly.png')
t0 = time.time()
clf = svm.SVC(kernel='rbf')
plot_learning_curve(
   clf,
    "SVM learning curve- kernel = rbf",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
print("SVM - rbf learning curve : ", time.time()-t0)
plt.savefig('images/SVM_Learning_curve_rbf.png')
plt.clf()
plt.close('images/SVM_Learning_curve_rbf.png')
plot_learning_curves(x_train,y_train,x_test,y_test,clf,scoring="accuracy")
plt.savefig('images/SVM_withoutcv.png')
plt.clf()
plt.close('images/SVM_withoutcv.png')

# #KNN ********************************************************
# *************************************************************
# *************************************************************
# K nearest neighbor dependency graphs**********************************
neighbors = range(1,50)
train_errors = []
test_errors = []
for i in neighbors:
    dtc = KNeighborsClassifier(n_neighbors=i,weights="distance")
    dtc.fit(x_train,y_train)
    pred_train = dtc.predict(x_train)
    pred_test = dtc.predict(x_test)
    train_errors.append(metrics.accuracy_score(y_train, pred_train))
    test_errors.append(metrics.accuracy_score(y_test, pred_test))
j = range(1,50)
plt.plot(j, train_errors, label='Training Accuracy')
plt.plot(j, test_errors, label='Testing Accuracy')
plt.xlabel('K nearest neighbor') # Label x-axis
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('images/K nearest neighbor-best k.png')
plt.clf()
plt.close('images/K nearest neighbor-best k.png')

# Learning curves ***********************************
#From previous section we understood that k =  is the best
t0 = time.time()
clf = KNeighborsClassifier(n_neighbors=7)
plot_learning_curve(
   clf,
    "KNN learning curve with 7 neighbors",
    x,
    y,
    cv=cv,
    n_jobs=4,
    scoring="accuracy",
)
print("KNN learning curve : ", time.time()-t0)
plt.savefig('images/KNN_Learning_curve_7.png')
plt.clf()
plt.close('images/KNN_Learning_curve_7.png')

plot_learning_curves(x_train,y_train,x_test,y_test,clf,scoring="accuracy")
plt.savefig('images/KNN_withoutcv.png')
plt.clf()
plt.close('images/KNN_withoutcv.png')
