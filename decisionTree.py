import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn import tree
import os
import io
import pydot

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

# predicted = cross_val_predict(regr, X_test, y_test, cv=10)

df = student.iloc[:, [26, 27]]

print(df.head())

features = df.columns[:8]

y = student["pass/fail"]
x = df[features]
dt = tree.DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(x, y)
tree.export_graphviz(dt, out_file="tree.dot")

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dotfile = io.StringIO()
tree.export_graphviz(dt, out_file=dotfile, feature_names=features)
(graph,) = pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_png("dtree.png")
os.system('dot -Tpng random.dot -o dtree.png')