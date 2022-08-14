import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from numpy.lib.function_base import average
from sklearn.ensemble import RandomForestClassifier
acc = []
df = pd.read_csv(r'C:\Users\ed069\Desktop\diabetes\User_Data.csv')
X = df.drop(columns=['Purchased','User ID','Gender'])
y = df['Purchased'].values
for i in range(1000):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state =i, stratify = y)
  model = RandomForestClassifier(n_estimators = 100, random_state = i)
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  # print('ground truth', y_test[0:10])
  # print('prediction: ',predictions[0:10])
  # print()
  # print(classification_report(y_test,predictions))
  print('Accuracy = ',accuracy_score(y_test,predictions))
  acc.append([accuracy_score(y_test,predictions)])
  for i in range(len(y_test)):
    if y_test[i] != predictions[i]:
      print(f'real value = {y_test[i]}, predict value = {predictions[i]}')
print(f'average accuracy = {average(acc)}')