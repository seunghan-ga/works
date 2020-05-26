import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# 모델 검정
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate  # 교차타당도
from sklearn.pipeline import make_pipeline  # 파이프라인 구축
from sklearn.model_selection import learning_curve, validation_curve  # 학습곡선, 검증곡선
from sklearn.model_selection import GridSearchCV  # 하이퍼 파라미터 튜닝
from sklearn.model_selection import cross_val_score  # 교차타당도
from sklearn.metrics import make_scorer

import seaborn as sns

sns.set(color_codes=True)
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model, model_from_json
from keras import regularizers

seed(10)
tf.compat.v1.set_random_seed(10)

data_dir = 'data/bearing_data'
merged_data = pd.DataFrame()
source_data = pd.read_csv('UniversalBank.csv')

print(source_data.keys())
X = source_data.drop(['ID', 'ZIPCode', 'PersonalLoan'], axis=1)
print(X.head())

y = source_data['PersonalLoan']

X['Education'] = X['Education'].replace([1, 2, 3], ['Under', 'Grad', 'Prof'])

X = pd.get_dummies(X[['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education',
                      'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard']],
                   columns=['Education'],
                   drop_first=True)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.9,
                     random_state=1,
                     stratify=y)
# X_train = X[1:9]
# y_train = y[1:9]

# # Source Data to DataFrame
#
# source_data = pd.read_csv('Averaged_BearingTest_Dataset.csv')
# source_data.index = pd.to_datetime(source_data['Index'], format='%Y-%m-%d %H:%M:%S')
# merged_data = source_data.drop(['Index'], axis=1)
#
# print("Dataset shape:", merged_data.shape)
# merged_data.head()
#
# # make dataset
# X_train = merged_data['2004-02-12 10:52:39': '2004-02-16 03:02:39']
# X_test = merged_data['2004-02-16 03:12:39':]
# # X_test = merged_data['2004-02-15 12:52:39':]
# X_train['Flag'] = 0
# X_test['Flag'] = 1
# y_train = X_train['Flag']
# y_test = X_test['Flag']
# X_train = X_train.drop(['Flag'], axis=1)
# X_test = X_test.drop(['Flag'], axis=1)
#
# X = pd.concat([X_train, X_test])
# y = pd.concat([y_train, y_test])
#
# X_train, X__test, y_train, y__test = train_test_split(X, y,
#                                                     test_size=0.3,
#                                                     random_state=0,
#                                                     stratify=y)
# X_train = X['2004-02-16 03:12:39':'2004-02-16 04:12:39']
# y_train = y['2004-02-16 03:12:39':'2004-02-16 04:12:39']
# X_test = X['2004-02-12 03:12:39':]
# y_test = y['2004-02-12 03:12:39':]
# pd.set_option('display.max_rows', 1000)
# print(X_test)
#
# # Dataset visualize
# fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
# ax.plot(X['Bearing 1'], label='Bearing 1', color='blue', linewidth=1)
# ax.plot(X['Bearing 2'], label='Bearing 2', color='red', linewidth=1)
# ax.plot(X['Bearing 3'], label='Bearing 3', color='green', linewidth=1)
# ax.plot(X['Bearing 4'], label='Bearing 4', color='black', linewidth=1)
# plt.legend(loc='lower left')
# ax.set_title('Bearing Sensor Training Data', fontsize=16)
# plt.show()

# Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# print("Training data shape:", X_train.shape)
# X_test = scaler.transform(X_test)
# print("Test data shape:", X_test.shape)
# scaler_filename = "scaler_data"
# joblib.dump(scaler, scaler_filename)


# 모델 구축
# tree = DecisionTreeClassifier(max_depth=None, criterion='entropy', random_state=1)
# forest = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=0)
# clf_labels = ['Decision tree', 'Random forest']
# all_clf = [tree, forest]

# # AUC 검정
# for clf, label in zip(all_clf, clf_labels):
#     scores = cross_val_score(estimator=clf,
#                              X=X_train,
#                              y=y_train,
#                              cv=10,
#                              scoring='roc_auc')
#     print("ROC AUC: %0.3f (+/- %0.3f) [%s]"
#           % (scores.mean(), scores.std(), label))
#
# # ROC 곡선
# colors = ['black', 'orange', 'blue', 'green']
# linestyles = [':', '--', '-.', '-']
#
# for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict_proba(X_test)[:, 1]
#     fpr, tpr, thresholds = roc_curve(y_true=y_test,
#                                      y_score=y_pred)
#     roc_auc = auc(x=fpr, y=tpr)
#     plt.plot(fpr, tpr,
#              color=clr,
#              linestyle=ls,
#              label='%s (auc = %0.3f)' % (label, roc_auc))
#
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1],
#          linestyle='--',
#          color='gray',
#          linewidth=2)
#
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])
# plt.grid(alpha=0.5)
# plt.xlabel('False positive rate (FPR)')
# plt.ylabel('True positive rate (TPR)')
#
# plt.show()

# 모델 검정
# forest.fit(X_train, y_train)
# y_pred = forest.predict(X_test)
# confmat = pd.DataFrame(confusion_matrix(y_test, y_pred),
#                        index=['True[0]', 'True[1]'],
#                        columns=['Predict[0]', 'Predict[1]'])

# print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())
# print('정확도: %.3f' % accuracy_score(y_test, y_pred))
# print('정밀도: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
# print('재현율: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
# print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

# # 기여도 분석
#
# feat_labels = X.columns
#
# importances = forest.feature_importances_
#
#
# indices = np.argsort(importances)[::-1]
#
# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30,
#                             feat_labels[indices[f]],
#                             importances[indices[f]]))
#
# plt.title('Feature Importance')
# plt.bar(range(X_train.shape[1]),
#         importances[indices],
#         align='center')
#
# plt.xticks(range(X_train.shape[1]),
#            feat_labels[indices], rotation=30)
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.show()

# 회귀 모델
# Regression 은 예측용으로 중요도 분류를 사용할 수 없음.

def rmsle(predicted_values, actual_values):
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)

    difference = log_predict - log_actual
    difference = np.square(difference)

    mean_difference = difference.mean()

    score = np.sqrt(mean_difference)

    return score


# RMSLE
# 과대평가 된 항목보다는 과소평가 된 항목에 페널티를 준다.
# 오차(Error)를 제곱(Square)해서 평균(Mean)한 값의 제곱근(Root)으로 값이 작을수록 정밀도가 높다.
# 0에 가까운 값이 나올수록 정밀도가 높은 값이다.
rmsle_scorer = make_scorer(rmsle)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# KFold 교차검증
# k_fold = KFold(n_splits=100, shuffle=True, random_state=0)

model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
# score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
# score = score.mean()
# print("Score= {0:.5f}".format(score))

model.fit(X_train, y_train)

# predictions = model.predict(X_test)
#
# print(predictions.shape)
#
# print('잘못 분류된 샘플 개수: %d' % (y_test != predictions).sum())
# print('정확도: %.3f' % accuracy_score(y_test, predictions))
# print('정밀도: %.3f' % precision_score(y_true=y_test, y_pred=predictions))
# print('재현율: %.3f' % recall_score(y_true=y_test, y_pred=predictions))
# print('F1: %.3f' % f1_score(y_true=y_test, y_pred=predictions))

# fig_reg, (ax1, ax2) = plt.subplots(ncols=2)
# fig_reg.set_size = (12, 5)
# sns.distplot(y_train, ax=ax1, bins=50)
# ax1.set(title="train")
# sns.distplot(predictions, ax=ax2, bins=50)
# ax2.set(title="test")
# plt.show()

feat_labels = X.columns

importances = model.feature_importances_


indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=30)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


