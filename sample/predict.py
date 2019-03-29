from sklearn import tree
import pandas as pd
import numpy as np

clf = tree.DecisionTreeClassifier()


# Data that is going to be retrieved from sensors is mock here.
df = pd.read_csv('datasets/symptoms.csv')

X_LABELS = df.iloc[:,:-1]
Y_LABELS = df.iloc[:,-1]

PARAM_COUNT = len(df.columns) - 1
# Exclude the last column for being label.

clf.fit(X_LABELS, Y_LABELS)

TEST_SET = np.tile(0, PARAM_COUNT)


for idx in range(PARAM_COUNT):
    TEST_SET[idx] = input('{param}: '
                          .format(param=df.columns[idx]))    


result = clf.predict([TEST_SET])


OUTPUT = {}
OUTPUT[0] = False
OUTPUT[1] = True

print('Fever present: {res}'
      .format(res=OUTPUT[result[0]]))

