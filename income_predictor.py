#Charles Petruzelli
#Income Predictor - Machine Learning EDA
#07/11/2018

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2

import os
os.chdir("C:/RScripts/EDA")

#Analyse the first few rows of the dataset
df = pd.read_csv('C:/RScripts/EDA/adult.csv')
df.head()
df.shape
df.describe()
df.info()

#Check the percentage and number of missing values in feature columns
for i in df.columns:
    non_value = df[i].isin(['?']).sum()
    if non_value > 0:
        print(i)
        print('{}'.format(float(non_value) / (df[i].shape[0]) * 100))
        print('\n')

#selecting all the rows without the '?' sign.
df = df[df['workclass'] != '?']
df = df[df['occupation'] != '?']
df = df[df['native_country'] != '?']

#Remove the 'fnlwgt' feature, which doesn't appear to be anything
df = df.drop('fnlwgt', axis=1)

#Correlation Plot
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot=True, cmap='magma', linecolor='white', linewidths=1)
plt.show()
#The correlation among features does not seem to be strong. We cannot drop the number of features, so we need to try another method.

plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['education'], palette = 'rainbow', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
#This plot tells us several interesting things:
    #Firstly, we can see that there are more people earning less than $50k than more than $50k
    #High school grads and some college are significantly higher in the less than $50k section.

plt.figure(figsize=(12,8))
sns.countplot(y = df['income'], hue = df['gender'], palette = 'summer', edgecolor = [(0,0,0), (0,0,0)])
plt.show()

#Here we can see the number of people earning more or less than 50K based on gender:
    #Males generate a higher count in both income groups
#We will now look into education, to see if that is a reason for the difference between income and gender

print("The number of men with each qualification")
print(df[df['gender'] == 'Male']['education'].value_counts())
plt.figure(figsize=(12,8))
sns.barplot(x = df[df['gender'] == 'Male']['education'].value_counts().values, y = df[df['gender'] == 'Male']['education'].value_counts().index, data = df)
plt.show()

#The above plot shows the count of men per different education types. High school graduates, some-college, and bachelors make up the biggest categories of men's education by volume.

#Now let's look at women:
("The number of women with each qualification")
print(df[df['gender'] == 'Female']['education'].value_counts())
plt.figure(figsize=(12,8))
sns.barplot(x = df[df['gender'] == 'Female']['education'].value_counts().values, y = df[df['gender'] == 'Female']['education'].value_counts().index, data = df)
plt.show()

#This plot shows that female education follows the same trend as males, however, the numbers are drastically different. The count of HS grads for men is 10,122, while for females it is 4,661. There is a much higher number of male HS grads compared to females, which may be a reason why more males earn less than $50K than females.
#This may also be the reason why there are more men making over $50K than females: there is a much larger number of men in each education category than when compared to the women.

#We can now look at the relationship category when compared to income
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['relationship'], palette = 'autumn', edgecolor = [(0,0,0), (0,0,0)])
plt.show()

#This plot shows that a majority of husbands earn more than $50K compared to other relationship groups. In the less than $50K group, husbands and not-in-family make up the majority.

#Now lets look at income with respect to occupation:
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['occupation'], palette = 'BuGn_d', edgecolor = [(0,0,0), (0,0,0)])
plt.show()

#This plot shows us several interesting figures.
    #First of all, people who's occupations are service, craft repair, professional specialty make up the majority of those in the less than $50K group.
    #Professional Specialty and executive managerial dominate the over $50K group.

#Now income with respect to race:
plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['race'], palette = 'Set3', edgecolor = [(0,0,0), (0,0,0)])
plt.show()

#Whites dominate both categories. A possible path to take this would be to see if the proportion of races in this sample represent the proportion of races in the United States.

#Income with respect to workclass:

plt.figure(figsize=(12,8))
sns.countplot(df['income'], hue = df['workclass'], palette = 'Dark2', edgecolor = [(0,0,0), (0,0,0)])
plt.show()
#There are more individuals in the private sector compared to all other workclasses, and in both income groups.

plt.figure(figsize=(12, 10))
sns.boxplot(x='income', y='age', data=df, hue='gender', palette = 'prism')
plt.show()

#### Data Cleaning #####
df['marital_status'].unique()
#Replace with unmarried and married to make the categorical variables easier to work with.
df['marital_status'] = df['marital_status'].replace(['Never-married', 'Married-civ-spouse', 'Widowed', 'Separated', 'Divorced',
                                  'Married-spouse-absent', 'Married-AF-spouse'], ['not married', 'married', 'not married',
                                   'not married', 'not married', 'not married', 'married'])
df.head()
#Converting categorical features to dummy variables
df = pd.get_dummies(df, columns=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender',  'native_country'], drop_first=True)
df.head()

#Splitting into features and labels, with features labeled as X and labels as Y
# Split the dataframe into features (X) and labels(y)
X = df.drop('income', axis=1)
y = df['income']

y = pd.get_dummies(y, columns=y, drop_first=True)
y = y.iloc[:,-1]
y.shape

#Split into Training and Testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

#Univariate Feature Selection
feature_select = SelectKBest(chi2, k = 8)  #finding the top 8 best features
feature_select.fit(X_train, y_train)

score_list = feature_select.scores_
top_features = X_train.columns

uni_features = list(zip(score_list, top_features))
print(sorted(uni_features, reverse=True)[0:8])

###### Random Forest #####
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X_train_1 = feature_select.transform(X_train)
X_test_1 = feature_select.transform(X_test)

#random forest classifier with n_estimators=10 (default)
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_1,y_train)

rf_pred = rf_clf.predict(X_test_1)

accu_rf = accuracy_score(y_test, rf_pred)
print('Accuracy is: ',accu_rf)

cm_1 = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_1, annot=True, fmt="d")
plt.show()
#~84% accuracy, howefver, the heat map shows that some of our predictions are incorrect. We can attempt to fix this by changing the number of top features and giving a trial and error method to improve the efficiency.

#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X_train_2 = feature_select.transform(X_train)
X_test_2 = feature_select.transform(X_test)


knn_clf = KNeighborsClassifier(n_neighbors=1)
knn_clf.fit(X_train_2,y_train)

knn_pred = knn_clf.predict(X_test_2)

accu_knn = accuracy_score(y_test, knn_pred)
print('Accuracy is: ',accu_knn)

cm_2 = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm_2, annot=True, fmt="d")
plt.show()
#~81% accurate.

accu_score = []

for k in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_2, y_train)
    prediction = knn.predict(X_test_2)
    accu_score.append(accuracy_score(prediction, y_test))

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(range(1, 50), accu_score)
plt.xlabel('K values')
plt.ylabel('Accuracy score')
plt.show()


X_train_3 = feature_select.transform(X_train)
X_test_3 = feature_select.transform(X_test)


knn_clf_1 = KNeighborsClassifier(n_neighbors=28)
knn_clf_1.fit(X_train_2,y_train)

knn_pred_1 = knn_clf_1.predict(X_test_2)

accu_knn_1 = accuracy_score(y_test, knn_pred_1)
print('Accuracy is: ',accu_knn_1)

cm_3 = confusion_matrix(y_test, knn_pred_1)
sns.heatmap(cm_3, annot=True, fmt="d", cmap='Dark2')
plt.show()

#K Fold Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

kfold = KFold(n_splits = 10, random_state = 5)

result = cross_val_score(rf, X_train_1, y_train, cv=kfold, scoring='accuracy')

print(result.mean())

#RFE With Cross-Validation and RF Classification
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_3 = RandomForestClassifier()
rfecv = RFECV(estimator=clf_rf_3, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X_train.columns[rfecv.support_])


import matplotlib.pyplot as plt

plt.figure(figsize = (10,8))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
