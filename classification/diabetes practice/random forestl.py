# import needed libraries
from numpy import average
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# use pandas to read the csv file
df = pd.read_csv('G:/vs code file/deep learning file/classification/diabetes practice/diabetes.csv') 

# X = 除了結果的其他8個值
X = df.drop(columns=['Outcome']) 
# y = Outcome(結果)
y = df['Outcome'].values 
acc = []
for i in range(100):
    #use train_test_split to split the ratio between the training data and the test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state =i+1, stratify = y) 
    # train the model using KNeighborsClassifier
    model = RandomForestClassifier(n_estimators = 100, random_state = 39)
    model.fit(X_train, y_train)
    # predict the X_test using trained model
    predictions = model.predict(X_test)
    # print report and accuracy
    acc.append(accuracy_score(y_test, predictions))
    print('Accuracy = ',accuracy_score(y_test,predictions))

print(average(acc))