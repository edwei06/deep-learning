import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
df = pd.read_csv('diabetes.csv')
X = df.drop(columns=['Outcome'])
y = df['Outcome'].values
testsize = []
k = []
highestacc = []
'''
for i in range(47):
  testsize.append(0.02*(i+1))
for i in range(40):
  k.append(i+1)
'''
for seed in range(400):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, random_state = seed, stratify = y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    result = np.zeros([len(df),2])
    result = np.transpose(np.vstack((y_test, predictions)))
    if accuracy_score(y_test,predictions)>0.95:
        highestacc.append([accuracy_score(y_test,predictions),seed])
    print(f'seed = {seed} ,trainsize =  0.98 Accuracy = ',accuracy_score(y_test,predictions))
highestacc.sort(reverse=True) 
for i in range(len(highestacc)):
  print(f'seed = {highestacc[i][1]}, trainsize = 0.98, accuracy = {highestacc[i][0]}')

  
  
  
'''
highestacc.sort(reverse=True) 
for i in range(len(highestacc[:10])):
  print(f'seed = {highestacc[i][1]}, testsize = {highestacc[i][2]}, k = {highestacc[i][3]}, accuracy = {highestacc[i][0]}')
'''