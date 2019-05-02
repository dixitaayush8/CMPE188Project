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

def f(row): #helper function that factorizes "pass/fail" column to 1/0 based on threshold of average out of 20. Threshold defined at 0.64.
   if row['average']/20 > 0.64:
       val = 1
   elif row['average']/20 <= 0.64:
       val = 0
   return val


def convertPstatus(row): #if parent's status is apart, set value to 0. Else, if they're together, set value to 1. 
   if row['Pstatus'] == 'A':
      val = 0
   elif row['Pstatus'] == 'T':
      val = 1
   return val

def main(): #main function to run machine learning computations and 
   data = pd.read_csv('student-por.csv') #read data from students who took a Portuguese class
   data.dropna(inplace=True) #drop any unnecessary NA values in the data

   df = data.iloc[:, [1, 2, 5, 6, 7, 13, 14, 20, 23, 24, 26, 27, 28, 29, 30, 31, 32]] #get sex, age, studytime, failures, higher, famrel, freetime, health, absences, ParentEdu, Talc, g1, g2, g3, Dalc, Walc, Pstatus, Medu, Fedu columns from dataset

   df['sex'] = data['sex'].map({'F': 0, 'M': 1}) #replace M/F in "sex" column to binary for data preprocessing
   df['higher'] = data['higher'].map({'no': 0, 'yes': 1}) #converts "yes/no" response to binary, higher represents whether user wants higher education or not

   df['average'] = df.iloc[:, -3:-1].astype(float).mean(axis=1) #create a new column in dataframe called "average". Averages G1, G2, and G3 columns. 
   df['ParentEdu'] = df.iloc[:, 4:6].astype(float).sum(axis=1) #sums up Medu + Fedu (mother and father education levels)
   df['Talc'] = df.iloc[:, 7:9].astype(float).sum(axis=1) #sums up Dalc and Walc
   df['PstatusNum'] = df.apply(convertPstatus, axis=1) #converts A/T to 0/1
   df['pass/fail'] = df.apply(f, axis=1) #applies "f" function to pass/fail column

   df.drop(['G1', 'G2', 'G3', 'Medu', 'Fedu', 'average', 'Dalc', 'Walc', 'Pstatus'], inplace=True, axis=1) #drop unnecessary columns after computation

   print(df.head()) #double checking the output so far

   print(len(df.columns))
   trainData, testData = train_test_split(df, test_size=0.3, random_state=1) #split data into training data and testing data

   passing = trainData[trainData.iloc[:, -1] == 1] #number of passing grade values
   failing = trainData[trainData.iloc[:, -1] == 0] #number of failing grade values

   print(failing.count()) 

   passingUpsample = resample(passing, replace=True, n_samples=310, random_state=1) #upsample data to number of passing matches number of failing

   trainUpsample = pd.concat([passingUpsample, failing])

   print("Training Data..........................................")
   print(trainUpsample.iloc[:, 0:11])

   X_train = trainUpsample.iloc[:, 0:11].values #take the first 11 columns
   y_train = trainUpsample.iloc[:, -1].values #take pass/fail column

   X_test = testData.iloc[:, 0:11].values #take first 11 columns
   y_test = testData.iloc[:, -1].values

   #nb = GaussianNB().fit(X_train, y_train)

   #----------------
   #checking for independence amongst features
   health = data['health']
   absences = data['absences']
   spearmanr_coefficient, p_value = spearmanr(health, absences)

   #since this would output -0.070, there is almost no correlation between the two features which is important for our purpose
   print("Spearman Rank correlation coefficient beween health and absences %0.3f" % (spearmanr_coefficient))

   sb.regplot(x='health', y='absences', data = data, scatter = True)

   # ----------------


   svm = SVC(kernel='linear').fit(X_train,y_train) #input model into SVM algorithm
   prediction = list(svm.predict(X_test))
   print(prediction)
   accuracy = round(svm.score(X_test,y_test) * 100, 2)
   print('Python SVM Accuracy: {}%'.format(accuracy))

   neighbors = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train) #input model into KNN algorithm
   accuracy = round(neighbors.score(X_test, y_test) * 100, 2)
   print('Python KNN Accuracy: {}%'.format(accuracy))

   #accuracy = round(nb.score(X_test, y_test) * 100, 2)
   #print('Python Accuracy for Naive Bayes: {}%'.format(accuracy))

   logit = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train) #input model into Logistic Regression
   accuracy = round(logit.score(X_test, y_test) * 100, 2)
   print('Python Logistic Regression Accuracy: {}%'.format(accuracy))
   plt.show()

   #take data from user 
   gender = str(input('Are you a male or female? (M or F)'))
   gender = 0 if gender=="F" else 1
   age = str(input('What is your age?'))
   age = int(age)
   Pstat = str(input('What is your parent\'s cohabitation status? (T - together or A - apart)'))
   Pstat = 0 if Pstat=="A" else 1
   Medu = str(input('What is your mother\'s education level? 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education'))
   Medu = int(Medu)
   Fedu = str(input('What is your father\'s education level? 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education'))
   Fedu = int(Fedu)
   ParentEdu = float(Medu + Fedu)
   studytime = str(input('How long do you study for? (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)'))
   studytime = int(studytime)
   failures = str(input('How many class failures have you had?'))
   failures = int(failures)
   higher = str(input('Do you want to get a higher education? (yes or no)'))
   higher = 0 if higher=="no" else 1
   famrel = str(input('How would you rate the quality of your relationship with your parents? (1 - bad, 5 - awesome)'))
   famrel = int(famrel)
   free_time = str(input('How much free time have you had after school? (1 is low, 5 is high)'))
   free_time = int(free_time)
   dalc = str(input('How often do you consume alcohol on a workday? (1 is low, 5 is high)'))
   dalc = int(dalc)
   walc = str(input('How often do you consume alcohol during the weekend? (1 is low, 5 is high)'))
   walc = int(walc)
   talc = float(dalc + walc)
   health = str(input('How would you rank your health? (1 is bad, 5 is awesome)'))
   health = int(health)
   absences = str(input('How many absences have you had during the school year so far? (93 is the max)'))
   absences = int(absences)
   #create one row of dataframe from user's inputs
   df_two = pd.DataFrame(columns=['sex', 'age', 'studytime','failures','higher','famrel','freetime','health','absences','ParentEdu','Talc'])
   print(df_two)
   df_two.loc[0] = pd.Series({'sex': gender, 'age': age,'studytime': studytime,'failures': failures,'higher': higher,'famrel': famrel,'freetime': free_time,'health': health,'absences': absences,'ParentEdu': ParentEdu,'Talc': talc})
   prediction = list(logit.predict(df_two.iloc[:, 0:11])) #use Logistic regression to predict user's chances of passing or failing based on alcoholic + behavioral habits
   print(prediction) #print prediction as list containing one of either 0 or 1

main()