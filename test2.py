import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#
import numpy as np
from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr
import seaborn as sb

from pylab import rcParams
import sklearn
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn import preprocessing

def average(score1, score2, score3):
   return (score1 + score2 + score3) / 3

def f(row):
   if row['average']/20 > 0.64:
       val = 1
   elif row['average']/20 <= 0.64:
       val = 0
   return val


def convertPstatus(row):
   if row['Pstatus'] == 'A':
      val = 0
   elif row['Pstatus'] == 'T':
      val = 1
   return val

def main():
   data = pd.read_csv('student-por.csv')
   data.dropna(inplace=True)

   df = data.iloc[:, [1, 2, 5, 6, 7, 13, 14, 20, 23, 24, 26, 27, 28, 29, 30, 31, 32]]

   df['sex'] = data['sex'].map({'F': 0, 'M': 1})
   df['higher'] = data['higher'].map({'no': 0, 'yes': 1})

   df['average'] = df.iloc[:, -3:-1].astype(float).mean(axis=1)
   df['ParentEdu'] = df.iloc[:, 4:6].astype(float).sum(axis=1)
   df['Talc'] = df.iloc[:, 7:9].astype(float).sum(axis=1)
   df['PstatusNum'] = df.apply(convertPstatus, axis=1)
   df['pass/fail'] = df.apply(f, axis=1)

   df.drop(['G1', 'G2', 'G3', 'Medu', 'Fedu', 'average', 'Dalc', 'Walc', 'Pstatus'], inplace=True, axis=1)

   print(df.head())

   print(len(df.columns))
   trainData, testData = train_test_split(df, test_size=0.3, random_state=1)

   passing = trainData[trainData.iloc[:, -1] == 1]
   failing = trainData[trainData.iloc[:, -1] == 0]

   print(failing.count())

   passingUpsample = resample(passing, replace=True, n_samples=310, random_state=1)

   trainUpsample = pd.concat([passingUpsample, failing])

   print("Training Data..........................................")
   print(trainUpsample.iloc[:, 0:11])

   X_train = trainUpsample.iloc[:, 0:11].values
   y_train = trainUpsample.iloc[:, -1].values

   X_test = testData.iloc[:, 0:11].values
   y_test = testData.iloc[:, -1].values

   nb = GaussianNB().fit(X_train, y_train)

   #----------------
   #checking for independence amongst features
   student_data = data.ix[:, (13,14)].values
   student_data_names = ['health', 'absences']

   #failure column
   y = data.ix[:,14].values

   health = data['health']
   absences = data['absences']
   spearmanr_coefficient, p_value = spearmanr(health, absences)

   #since this would output -0.070, there is almost no correlation between the two features which is important for our purpose
   print("Spearmanr Rank correlation coefficient beween health and absences %0.3f" % (spearmanr_coefficient))

   sb.regplot(x='health', y='absences', data = data, scatter = True)

   # ----------------


   svm = SVC(kernel='linear').fit(X_train,y_train)
   prediction = list(svm.predict(X_test))
   print(prediction)
   accuracy = round(svm.score(X_test,y_test) * 100, 2)
   print('Python SVM Accuracy: {}%'.format(accuracy))

   neighbors = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
   accuracy = round(neighbors.score(X_test, y_test) * 100, 2)
   print('Python KNN Accuracy: {}%'.format(accuracy))

   accuracy = round(nb.score(X_test, y_test) * 100, 2)
   print('Python Accuracy for Naive Bayes: {}%'.format(accuracy))

   logit = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
   accuracy = round(logit.score(X_test, y_test) * 100, 2)
   print('Python Logistic Regression Accuracy: {}%'.format(accuracy))
   print(plt.show())
main()