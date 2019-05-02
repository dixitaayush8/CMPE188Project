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
    data = pd.read_csv('student-por.csv')
    data.dropna(inplace=True)

    data['sex'] = data['sex'].map({'F': 0, 'M': 1})
    data['higher'] = data['higher'].map({'no': 0, 'yes': 1})

    df = data.iloc[:, [1, 2, 14, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32]]

    print(df.head(5))

    df['average'] = df.iloc[:, -3:-1].astype(float).mean(axis=1)
    df['pass/fail'] = df.apply(f, axis=1)

    df.drop(['G1', 'G2', 'G3'], inplace=True, axis=1)
    print(df.head(100))

    print(len(df.columns))
    trainData, testData = train_test_split(df, test_size=0.3, random_state=1)

    passing = trainData[trainData.iloc[:, -1] == 1]
    failing = trainData[trainData.iloc[:, -1] == 0]

    passingUpsample = resample(passing, replace=True, n_samples=320, random_state=1)
    trainUpsample = pd.concat([passingUpsample, failing])

    X_train = trainUpsample.iloc[:, 0:10].values
    y_train = trainUpsample.iloc[:, -1].values

    X_test = testData.iloc[:, 0:10].values
    y_test = testData.iloc[:, -1].values

    # nb = GaussianNB().fit(X_train, y_train)
    # prediction = list(nb.predict(X_test))
    # accuracy = round(nb.score(X_test, y_test) * 100, 2)
    # print('Python Accuracy for Naive Bayes: {}%'.format(accuracy))

    svm = SVC(kernel='linear').fit(X_train,y_train)
    prediction = list(svm.predict(X_test))
    accuracy = round(svm.score(X_test,y_test) * 100, 2)
    print('Python SVM Accuracy: {}%'.format(accuracy))

    # neighbors = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    # accuracy = round(neighbors.score(X_test, y_test) * 100, 2)
    # print('Python KNN Accuracy: {}%'.format(accuracy))

    logit = LogisticRegression(random_state=0, solver='liblinear').fit(X_train, y_train)
    accuracy = round(logit.score(X_test, y_test) * 100, 2)
    print('Python Logistic Regression Accuracy: {}%'.format(accuracy))

main()