## Library Import


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (12, 10)
import seaborn as sns

sns.set_style('whitegrid')

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier, plot_importance

#Data Load

# # Titanic dataset load


train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
sub = pd.read_csv("./input/gender_submission.csv")
train.head()

print(train.describe(include='all'))

test.head()

test.describe(include='all')

sub.head()

sub.describe()

#EDA

#Correlations

corr = train.drop('PassengerId', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='YlOrBr')
plt.show()

sns.countplot(train['Pclass'], hue=train['Survived'])
plt.show()

plt.hist(train[train["Survived"] == 1]['Fare'], label="Yes", alpha=0.7)
plt.hist(train[train["Survived"] == 0]['Fare'], label="No", alpha=0.7)
plt.legend(title='Survived')
plt.xlabel("Fare")
plt.show()

#train, test data 합치기

Survival = train.Survived# Label for training set
full = pd.concat([train.drop('Survived', axis=1), test])
full.describe(include="all")

type(Survival)

#Feature Engineering

full1 = full.drop(['PassengerId', 'Cabin', 'Ticket'], axis=1)
full1.describe(include="all")

pclass = pd.get_dummies(full1['Pclass'], prefix="Pclass_")
pclass.head()

full1['Sex_'] = np.where(full1.Sex == 'male', 1, 0)
full1.head()

Embarked = pd.get_dummies(full1['Embarked'], prefix="Embarked_")
Embarked.head()

full2 = pd.concat([full1, pclass, Embarked], axis=1)
full2.head()

full3 = full2.drop(["Pclass", "Sex", "Embarked"], axis=1)
full3.describe(include="all")

Title = pd.get_dummies(full.Name.map(lambda x: x.split(',')[1].split('.')[0].split()[-1]))
Title.head()

full3['FamilySize'] = full3.SibSp + full3.Parch + 1
full3['Single'] = np.where((full3.SibSp + full3.Parch) == 0, 1, 0)
full4 = pd.concat([full3, Title], axis=1)
full4.drop('Name', axis=1, inplace=True)
full4.head()

#Correlations new feature

plt.figure(figsize=(26, 24))
sns.heatmap(full4.corr(), annot=True, cmap="YlOrBr")

full5 = full4.loc[:, 'Age':'Single']# or: full4.iloc[:,0:13]
full5.head()

plt.figure(figsize=(16, 14))
sns.heatmap(full5.corr(), annot=True, cmap="YlOrBr")

full6 = full5.drop(['SibSp', 'Parch', 'Single'], axis=1)
full6.head()

plt.figure(figsize=(16, 14))
sns.heatmap(full6.corr(), annot=True, cmap="YlOrBr")

#train, test data로 다시 분리

train_full = full6.iloc[:891]
test_full = full6.iloc[891:]
train_full.describe(include="all")

test_full.describe(include="all")

#Impute the missing Values

train_age_imputer = SimpleImputer()
train_imputed = train_full.copy()
train_imputed['Age_'] = train_age_imputer.fit_transform(train_full.iloc[:, 0:1])
train_imputed['Fare_'] = train_imputed['Fare']
train_imputed.drop(['Age', 'Fare'], axis=1, inplace=True)
train_imputed.head()

test_age_imputer = SimpleImputer()
test_fare_imputer = SimpleImputer()

test_imputed = test_full.copy()
test_imputed['Age_'] = test_age_imputer.fit_transform(test_full.iloc[:, 0:1])
test_imputed['Fare_'] = test_age_imputer.fit_transform(test_full.iloc[:, 1:2])

test_imputed.drop(["Age", "Fare"], axis=1, inplace=True)
test_imputed.head()

train_imputed.describe(include="all")

test_imputed.describe(include="all")

#EDA (새로운 feature 추가)

sns.countplot(Survival, palette="coolwarm")

# Survival by Sex
sns.countplot(Survival, hue=train_imputed["Sex_"], palette="coolwarm")

# Survival by PClass
sns.countplot(Survival, hue=train["Pclass"], palette="coolwarm")

# Survival by Age
plt.hist(train_imputed[Survival == 1]['Age_'], label="Survived", alpha=0.7)
plt.hist(train_imputed[Survival == 0]['Age_'], label="Not Survived", alpha=0.7)
plt.legend()
plt.xlabel("Age_Imputed")

# Survival by Fare
plt.hist(train_imputed[Survival == 1]['Fare_'], label="Survived", alpha=0.5)
plt.hist(train_imputed[Survival == 0]['Fare_'], label="Not Survived", alpha=0.5)
plt.legend()
plt.xlabel("Fare")
plt.title("Imputed Fare")

# Survival by Embarked
sns.countplot(Survival, data=train, hue="Embarked", palette="coolwarm")

# Survival by Family size
sns.countplot('FamilySize', data=train_imputed, palette="coolwarm", hue=Survival)
plt.legend(loc=1)

#Modeling

kfold = KFold(n_splits=10, random_state=1, shuffle=True)
kfold

accuracy = {}

# Export dataset
# train_imputed.to_csv('train_data.csv')
# Survival.to_csv('train_label.csv')


#Gausian Naive Bayes

m1_nb = GaussianNB()
accuracy['Gaussian Naive Bayes'] = np.mean(
cross_val_score(m1_nb, train_imputed, Survival, scoring="accuracy", cv=kfold))

#Logistic Regression

m2_log = LogisticRegression(solver='newton-cg')# 'lbfgs', 'sag' failed to converge

accuracy['Logistic Regression'] = np.mean(
cross_val_score(m2_log, train_imputed, Survival, scoring="accuracy", cv=kfold))

#K-Nearest Neighbors

m3_knn_5 = KNeighborsClassifier(n_neighbors=5)
m3_knn_10 = KNeighborsClassifier(n_neighbors=10)
m3_knn_30 = KNeighborsClassifier(n_neighbors=30)

accuracy['KNN'] = np.max([np.mean(cross_val_score(m3_knn_5, train_imputed, Survival, scoring="accuracy", cv=kfold)),
np.mean(cross_val_score(m3_knn_10, train_imputed, Survival, scoring="accuracy", cv=kfold)),
np.mean(cross_val_score(m3_knn_30, train_imputed, Survival, scoring="accuracy", cv=kfold))])

#Random Forests

m4_rf = RandomForestClassifier(n_estimators=300)

accuracy['Random Forest'] = np.mean(cross_val_score(m4_rf, train_imputed, Survival, scoring="accuracy", cv=kfold))

#Support Vector Machine

m5_svc = SVC(gamma='scale')

accuracy['SVM'] = np.mean(cross_val_score(m5_svc, train_imputed, Survival, scoring="accuracy", cv=kfold))

#Gradient Boosting

m6_gb = XGBClassifier(max_depth=10, n_estimators=200, learning_rate=0.01)

accuracy['Gradient Boosting'] = np.mean(cross_val_score(m6_gb, train_imputed, Survival, scoring="accuracy", cv=kfold))

#IsolationForest

m7_if = IsolationForest(n_estimators=100, random_state=np.random.RandomState(42))

accuracy['Isolation Forest'] = np.mean(cross_val_score(m7_if, train_imputed, Survival, scoring="accuracy", cv=kfold))

#AdamClass

m8_ab = AdaBoostClassifier(n_estimators=300, learning_rate=0.05)

accuracy['Adam Class'] = np.mean(cross_val_score(m8_ab, train_imputed, Survival, scoring="accuracy", cv=kfold))

#Check Each Model's Accuracy

print(accuracy)

max_accuracy = max(accuracy, key=accuracy.get)
print(max_accuracy, '\taccuracy:', accuracy[max_accuracy])

# Important Feature?

# RandomForestClassifier
m4_rf.fit(train_imputed, Survival)
rf_predictions = m4_rf.predict(test_imputed)
m4_rf.feature_importances_

Survival.shape

train_imputed.info()

m4_rf.predict_proba(train_imputed)

feat_importances = pd.Series(m4_rf.feature_importances_, index=train_imputed.columns)
feat_importances.nlargest(5).plot(kind='barh')
feat_importances

# XGBoost
m6_gb.fit(train_imputed, Survival)
m6_gb.feature_importances_

feat_importances = pd.Series(m6_gb.feature_importances_, index=train_imputed.columns)
feat_importances.nlargest(5).plot(kind='barh')

plot_importance(m6_gb)

# #Tune Hyper parameters
# 
# param_grid = {'max_depth': [1, 3, 5, 10, 15], 'n_estimators': [50, 100, 200, 500, 1000],
# 'learning_rate': [1, 0.1, 0.01, 0.001, 0.0001]}
# grid = GridSearchCV(XGBClassifier(), param_grid, cv=kfold)
# grid.fit(train_imputed[['Fare_', 'Age_', 'FamilySize', 'Sex_']], Survival)
# grid.best_params_
# 
# gb = XGBClassifier(max_depth=10, n_estimators=200, learning_rate=0.01)
# np.mean(
# cross_val_score(gb, train_imputed[['Fare_', 'Age_', 'FamilySize', 'Sex_']], Survival, scoring='accuracy', cv=kfold))
# 
# gb.fit(train_imputed[['Fare_', 'Age_', 'FamilySize', 'Sex_']], Survival)
# predictions = gb.predict(test_imputed[['Fare_', 'Age_', 'FamilySize', 'Sex_']])
# 
# result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
# 
# print(result)
