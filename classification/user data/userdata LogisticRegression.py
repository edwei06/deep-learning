import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.lib.function_base import average
from sklearn.linear_model import LogisticRegression
acc = []
df = pd.read_csv(r'C:\Users\ed069\Desktop\diabetes\User_Data.csv')
X = df.drop(columns=['Purchased','User ID','Gender'])
y = df['Purchased'].values
for i in range(2**15):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state =i, stratify = y)
  model = LogisticRegression()
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