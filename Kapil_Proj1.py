# -*- coding: utf-8 -*-
"""

@author: Kapil Tripathi
"""
# Reading the dataset
import pandas as pd
df = pd.read_csv("Proj1.csv")
df.shape
df.head
df.dtypes
list(df)
df.isnull().sum()
df.isna().sum() # There is NA values in the data set i.e. in Sample ID, So We will drop it.
df.describe()
df.drop(['Sl No','Sample ID','Age'], axis=1, inplace=True)
df

# LabelEncoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Gender'] = LE.fit_transform(df['Gender'])

corr = df.corr()
rela = corr['Gender'].sort_values(ascending = False)
rela

# Splitting the X and Y(Target)variables
X = df.drop(['Gender'], axis=1)

# X = df.drop(['Gender','right canine width intraoral','left canine width intraoral','left canine index intraoral'], axis=1)

Y = df['Gender'] # Here Gender is my Target Variable.

# Scatter Plot
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df.plot.scatter(x="left canine width casts", y="Gender")
df.plot.scatter(x="left canine width intraoral", y="Gender")
df.plot.scatter(x="right canine width casts", y="Gender")
df.plot.scatter(x="right canine width intraoral", y="Gender")
df.plot.scatter(x="left canine index casts", y="Gender")
df.plot.scatter(x="left canine index intraoral", y="Gender")
df.plot.scatter(x="inter canine distance intraoral", y="Gender")
df.plot.scatter(x="right canine index casts", y="Gender")
df.plot.scatter(x="right canine index intra oral", y="Gender")
df.plot.scatter(x="intercanine distance casts", y="Gender")

plt.subplots(figsize=(8,8))
sns.heatmap(corr, annot=True)

# Normalization
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(0,1))
X = Scaler.fit_transform(X)
pd.DataFrame(X)

# Splitting Train and Test DataSet
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, stratify=Y, random_state=1)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

# Model Development

# 1. LogisticRegression
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X_train,Y_train)
LogReg.intercept_
LogReg.coef_
Y_pred_train = LogReg.predict(X_train)
Y_pred_test = LogReg.predict(X_test)

# Evaluting Model Performance

from sklearn.metrics import confusion_matrix,accuracy_score
CM = confusion_matrix(Y_test,Y_pred_test)
CM
Accuracy_Score = accuracy_score(Y_test,Y_pred_test)*100
Accuracy_Score.round(3)


# 2. KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5, p=2)
KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)

# Evaluting Model Performance

from sklearn.metrics import confusion_matrix,accuracy_score
CM = confusion_matrix(Y_test,Y_pred_test)
CM
Accuracy_Score = accuracy_score(Y_test,Y_pred_test)*100
Accuracy_Score.round(3)

from sklearn.model_selection import GridSearchCV
samples = {'n_neighbors': [5,6,7,8,9,10,11,12,13,14,15], 'p':[2]}
grid = GridSearchCV(KNeighborsClassifier(),param_grid=samples,scoring = 'accuracy')
KNN_grid = grid.fit(X_train,Y_train)
KNN_grid.fit(X_test,Y_test)

KNN_best_score = KNN_grid.best_score_*100
KNN_best_score.round(3)
KNN_grid.best_params_


# 3. Naive Bayes MultinomialNB
from sklearn.naive_bayes import MultinomialNB
MNB= MultinomialNB()
MNB.fit(X_train,Y_train)
Y_pred = MNB.predict(X_test)

# Evaluting Model Performance
from sklearn.metrics import accuracy_score
CM = confusion_matrix(Y_test,Y_pred_test)
CM
Accuracy_Score = accuracy_score(Y_test,Y_pred)*100
Accuracy_Score.round(3)

# 4. Naive Bayes GaussianNB
from sklearn.naive_bayes import GaussianNB
GNB= GaussianNB()
GNB.fit(X_train,Y_train)
Y_pred = GNB.predict(X_test)

# Evaluting Model Performance
from sklearn.metrics import accuracy_score
CM = confusion_matrix(Y_test,Y_pred_test)
CM
Accuracy_Score = accuracy_score(Y_test,Y_pred)*100
Accuracy_Score.round(3)

# 5. DecisionTreeClassifier (criterion='gini' or 'entropy')

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X_train,Y_train)
Y_Pred = DT.predict(X_test)

DT.tree_.node_count
DT.tree_.max_depth

# Evaluting Model Performance
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test,Y_Pred)
Accuracy_Score = accuracy_score(Y_test,Y_Pred)*100
Accuracy_Score.round(3)

# Tunning the Model by GridSearch Method
from sklearn.model_selection import GridSearchCV
samples = {'criterion':['entropy','gini'],'max_depth': [1,2,3,4,5,6,7]}
grid = GridSearchCV(DecisionTreeClassifier(),param_grid=samples,scoring = 'accuracy',cv=10)
DT_grid = grid.fit(X_train,Y_train)
DT_grid.fit(X_test,Y_test)

DT_best_score = DT_grid.best_score_*100
DT_best_score.round(3)
DT_grid.best_params_

DT_grid.best_estimator_

# 6. RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,min_samples_split=25, max_depth=7, max_features=2)
RFC.fit(X_train,Y_train)
Y_Pred = RFC.predict(X_test)

# Evaluting Model Performance
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test,Y_Pred)
Accuracy_Score = accuracy_score(Y_test,Y_Pred)*100
Accuracy_Score.round(3)

# Tunning the Model by GridSearch Method
from sklearn.model_selection import GridSearchCV
samples = {'n_estimators':[10,20,30,40],'min_samples_split':[25,50,75,100],'max_depth':[1,2,3,4,5,6,7],'max_features':[2]}
grid = GridSearchCV( RandomForestClassifier(),param_grid=samples,scoring = 'accuracy',cv=10)
RFC_grid = grid.fit(X_train,Y_train)
RFC_grid.fit(X_test,Y_test)

RFC_best_score = DT_grid.best_score_*100
RFC_best_score.round(3)
RFC_grid.best_params_

