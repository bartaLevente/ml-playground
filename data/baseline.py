from sklearn.dummy import DummyClassifier
import pandas as pd
dummy = DummyClassifier()

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

X_train = train.iloc[:, 1]
y_train = train.iloc[:, -1]

print(X_train.head())
print(y_train.head())

dummy.fit(X_train, y_train)
X_test = test.iloc[:, 1]
y_test = test.iloc[:, -1]

dummy.fit(X_train, y_train)
print("train score: ", dummy.score(X_train, y_train))
print("test score: ", dummy.score(X_test, y_test))

