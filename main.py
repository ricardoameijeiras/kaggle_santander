import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

route = 'E:\Downloads\santander-customer-transaction-prediction'

train_set = pd.read_csv(os.path.join(route, 'train.csv'), sep=',')
test_set = pd.read_csv(os.path.join(route, 'test.csv'), sep=',')

print(train_set.shape)
print(test_set.shape)

print(train_set.describe())

train_set.drop(['ID_code'], inplace=True, axis=1)
test_set.drop(['ID_code'], inplace=True, axis=1)

plt.figure(figsize = (15, 8))
for i in range(1, 9):
    plt.subplot(2, 4, i)
    sns.boxplot(data = train_set.iloc[:, i])
    plt.title(train_set.columns[i])

plt.show()

train_set = train_set[(np.abs(stats.zscore(train_set)) < 3).all(axis=1)]
print(train_set.shape)

plt.figure(figsize = (15, 8))
for i in range(1, 9):
    plt.subplot(2, 4, i)
    sns.boxplot(data = train_set.iloc[:, i])
    plt.title(train_set.columns[i])

plt.show()

y = train_set['target']
X = train_set.drop(['target'], axis=1)
print(y.shape)
print(X.shape)

scaler = MinMaxScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1))
X = scaler.fit_transform(X)

cv = KFold(n_splits=12, random_state=42, shuffle=True)

model2 = LogisticRegression()

# scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
# print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state = 42)

model2.fit(xtrain, ytrain)

predicted2 = model2.predict_proba(xtrain)[:,1].round()
train_score2 = accuracy_score(predicted2, ytrain)*100

light = LGBMClassifier(n_jobs=-1)
light.fit(xtrain, ytrain, eval_metric = 'auc', eval_set = (xtest, ytest), verbose = 10, early_stopping_rounds = 100)

predicted = light.predict_proba(ytrain).round()
train_score = accuracy_score(predicted, ytrain)*100
print("Accuracy using LightGBM on training data is {} %".format(train_score))
print("Roc auc score using LightGBM on training data is {} %".format(roc_auc_score(predicted, ytrain)))

predicted = model2.predict_proba(ytrain)[:,1].round()
train_score = accuracy_score(predicted, ytest)*100

print(predicted.shape)
# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(ytest, predicted, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(ytest, predicted2, pos_label=1)



# roc curve for tpr = fpr
random_probs = [0 for i in range(len(ytest))]
p_fpr, p_tpr, _ = roc_curve(ytest, random_probs, pos_label=1)

plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='LGBM')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Logistic regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();