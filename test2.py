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

from pylab import rcParams
import seaborn as sb
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

def main():
   data = pd.read_csv('student-por.csv')
   data.dropna(inplace=True)

   df = data.iloc[:, [13, 14, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
   print(df.head())
   df['average'] = df.iloc[:, 8:11].astype(float).mean(axis=1)
   df['pass/fail'] = df.apply(f,axis=1)

   print(df.head())
   print(len(df.columns))
   trainData, testData = train_test_split(df, test_size=0.3, random_state=1)
   X_train = trainData.iloc[:, 0:7]
   y_train = trainData.iloc[:, 12]

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
   print(“Spearmanr Rank correlation coefficient beween health and absences %0.3f” % (spearmanr_coefficient))

   sb.regplot(x='health', y='absences', data = data, scatter = True)

   # ----------------
   nb = GaussianNB().fit(X_train, y_train)
   X_test = testData.iloc[:, 0:7]
   y_test = testData.iloc[:, 12]
   prediction = list(nb.predict(X_test))
   print(prediction)

   svm = SVC(kernel='linear').fit(X_train,y_train)
   X_test = testData.iloc[:, 0:7]
   y_test = testData.iloc[:, 12]
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