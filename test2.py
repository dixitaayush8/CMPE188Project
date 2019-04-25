import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def average(score1, score2, score3):
    return (score1 + score2 + score3) / 3

def f(row):
    if row['average']/20 > 0.64:
        val = 1
    elif row['average']/20 <= 0.64:
        val = 0
    return val


def main():
    data = pd.read_csv('student-mat.csv')
    data.dropna(inplace=True)

    df = data.iloc[:, [13, 14, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
    #print(df.head())
    df['average'] = df.iloc[:, 8:11].astype(float).mean(axis=1)
    df['pass/fail'] = df.apply(f,axis=1)
    #df.loc[df['average']/20 > 0.64, 'pass/fail'] == 1
    #df.loc[df['average']/20 <= 0.64, 'pass/fail'] == 0
    #df['average'] = df['average'].astype(int)
    #df['average'] = df['average'] + 1
    print(df.head())
    #df.loc[df.index[11], 'average'] = average(df[8], df[9],df[10])
    print(len(df.columns))
    trainData, testData = train_test_split(df, test_size=0.3, random_state=1)
    X_train = trainData.iloc[:, 0:7]
    y_train = trainData.iloc[:, 12]

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

main()