# import needed libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from numpy import average

# use pandas to read the csv file
df = pd.read_csv('G:/vs code file/deep learning file/classification/diabetes practice/diabetes.csv') 

# X = 除了結果的其他8個值
X = df.drop(columns=['Outcome']) 
# y = Outcome(結果)
y = df['Outcome'].values 
acc = []
for i in range(1000):
#use train_test_split to split the ratio between the training data and the test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state =i+1, stratify = y) 
    # train the model using KNeighborsClassifier
    model = tree.DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)
    # predict the X_test using trained model
    predictions = model.predict(X_test)
    # print report and accuracy
    acc.append(accuracy_score(y_test, predictions))
    print('Accuracy = ',accuracy_score(y_test,predictions))

print(f'average acc = {average(acc)}')