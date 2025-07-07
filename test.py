import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import normalize

dataset = pd.read_csv("Dataset/lifestyle_dataset.csv")
print(dataset.head())
print(np.unique(dataset['CVD']))

columns = ['Eating_Habits','Physical_Activity','BMI','Stress','Sleep','Smoking','Alcohol','Gender',
           'CVD','DM','CKD','COPD','PCOD','HLD','HTN','LC','AD']

le = LabelEncoder()
for i in range(len(columns)):
    dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))


dataset = dataset.values
X = dataset[:,1:10]
#X = normalize(X)
Y1 = dataset[:,10]
Y2 = dataset[:,11]
Y3 = dataset[:,12]
Y4 = dataset[:,13]
Y5 = dataset[:,14]
Y6 = dataset[:,15]
Y7 = dataset[:,16]
Y8 = dataset[:,17]
Y9 = dataset[:,18]
print(Y1)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y1 = Y1[indices]
Y2 = Y2[indices]
Y3 = Y3[indices]
Y4 = Y4[indices]
Y5 = Y5[indices]
Y6 = Y6[indices]
Y7 = Y7[indices]
Y8 = Y8[indices]
Y9 = Y9[indices]


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y1, test_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y2, test_size=0.2)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y3, test_size=0.2)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, Y4, test_size=0.2)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X, Y5, test_size=0.2)
X_train6, X_test6, y_train6, y_test6 = train_test_split(X, Y6, test_size=0.2)
X_train7, X_test7, y_train7, y_test7 = train_test_split(X, Y7, test_size=0.2)
X_train8, X_test8, y_train8, y_test8 = train_test_split(X, Y8, test_size=0.2)
X_train9, X_test9, y_train9, y_test9 = train_test_split(X, Y9, test_size=0.2)

cls1 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls1.fit(X,Y1)
predict1 = cls1.predict(X_test1) 
svm_acc1 = accuracy_score(y_test1,predict1)*100
print(svm_acc1)

cls2 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls2.fit(X,Y2)
predict2 = cls2.predict(X_test2) 
svm_acc2 = accuracy_score(y_test2,predict2)*100
print(svm_acc2)

cls3 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls3.fit(X,Y3)
predict3 = cls3.predict(X_test3) 
svm_acc3 = accuracy_score(y_test3,predict3)*100
print(svm_acc3)

cls4 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls4.fit(X,Y4)
predict4 = cls4.predict(X_test4) 
svm_acc4 = accuracy_score(y_test4,predict4)*100
print(svm_acc4)

cls5 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls5.fit(X,Y5)
predict5 = cls5.predict(X_test5) 
svm_acc5 = accuracy_score(y_test5,predict5)*100
print(svm_acc5)

cls6 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls6.fit(X,Y6)
predict6 = cls6.predict(X_test6) 
svm_acc6 = accuracy_score(y_test6,predict6)*100
print(svm_acc6)

cls7 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls7.fit(X,Y7)
predict7 = cls7.predict(X_test7) 
svm_acc7 = accuracy_score(y_test7,predict7)*100
print(svm_acc7)

cls8 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls8.fit(X,Y8)
predict8 = cls8.predict(X_test8) 
svm_acc8 = accuracy_score(y_test8,predict8)*100
print(svm_acc8)

cls9 = svm.SVC(C=3.0,gamma='scale',kernel = 'rbf', random_state = 42, class_weight="balanced")
cls9.fit(X,Y9)
predict9 = cls9.predict(X_test9) 
svm_acc9 = accuracy_score(y_test9,predict9)*100
print(svm_acc9)









