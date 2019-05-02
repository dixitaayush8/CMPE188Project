import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

def f(row):
    if row['average']/20 > 0.64:
        val = 1
    elif row['average']/20 <= 0.64:
        val = 0
    return val

student = pd.read_csv("./student-por.csv")

student.dropna(inplace=True)

student['average'] = student.iloc[:, 30:32].astype(float).mean(axis=1)
student['pass/fail'] = student.apply(f, axis=1)

X,y = student.iloc[:,:-5],student.iloc[:,-1]

le=LabelEncoder()
for col in X.columns.values:
    if X[col].dtypes=='object':
        le.fit(X[col].values)
        X[col]=le.transform(X[col])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

predicted = cross_val_predict(regr, X_test, y_test, cv=10)



# box plot
gradeData = student.iloc[:,-2]

plt.title("Boxplot for Average Grade")

boxplotData = "Median: " + str(np.median(gradeData)) + "\n" \
          + "Max: " + str(np.max(gradeData)) + "\n" \
          + "Min: " + str(np.min(gradeData)) + "\n" \
          + "Q3: " + str(np.percentile(gradeData, 75)) + "\n" \
          + "Q1: " + str(np.percentile(gradeData, 25)) + "\n" \
          + "IQR: " + str(np.percentile(gradeData, 75) - np.percentile(gradeData, 25))

gradeData.plot(kind='box')
plt.show()

print(boxplotData)