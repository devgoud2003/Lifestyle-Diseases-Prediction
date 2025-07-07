import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from flask import Flask, render_template, request, redirect, Response
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



app = Flask(__name__)
app.secret_key = 'dropboxapp1234'
global cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9
global svm_acc1, svm_acc2, svm_acc3, svm_acc4, svm_acc5, svm_acc6, svm_acc7, svm_acc8, svm_acc9 
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()
le7 = LabelEncoder()
le8 = LabelEncoder()


le9 = LabelEncoder()
le10 = LabelEncoder()
le11 = LabelEncoder()
le12 = LabelEncoder()
le13 = LabelEncoder()
le14 = LabelEncoder()
le15 = LabelEncoder()
le16 = LabelEncoder()
le17 = LabelEncoder()

@app.route("/TrainML")
def TrainML():
    global svm_acc1, svm_acc2, svm_acc3, svm_acc4, svm_acc5, svm_acc6, svm_acc7, svm_acc8, svm_acc9       
    global cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17
    dataset = pd.read_csv("Dataset/lifestyle_dataset.csv")
    print(dataset.head())
    print(np.unique(dataset['CVD']))

    columns = ['Eating_Habits','Physical_Activity','BMI','Stress','Sleep','Smoking','Alcohol','Gender',
           'CVD','DM','CKD','COPD','PCOD','HLD','HTN','LC','AD']

    dataset[columns[0]] = pd.Series(le1.fit_transform(dataset[columns[0]].astype(str)))
    dataset[columns[1]] = pd.Series(le2.fit_transform(dataset[columns[1]].astype(str)))
    dataset[columns[2]] = pd.Series(le3.fit_transform(dataset[columns[2]].astype(str)))
    dataset[columns[3]] = pd.Series(le4.fit_transform(dataset[columns[3]].astype(str)))
    dataset[columns[4]] = pd.Series(le5.fit_transform(dataset[columns[4]].astype(str)))
    dataset[columns[5]] = pd.Series(le6.fit_transform(dataset[columns[5]].astype(str)))
    dataset[columns[6]] = pd.Series(le7.fit_transform(dataset[columns[6]].astype(str)))
    dataset[columns[7]] = pd.Series(le8.fit_transform(dataset[columns[7]].astype(str)))

    dataset[columns[8]] = pd.Series(le9.fit_transform(dataset[columns[8]].astype(str)))
    dataset[columns[9]] = pd.Series(le10.fit_transform(dataset[columns[9]].astype(str)))
    dataset[columns[10]] = pd.Series(le11.fit_transform(dataset[columns[10]].astype(str)))
    dataset[columns[11]] = pd.Series(le12.fit_transform(dataset[columns[11]].astype(str)))
    dataset[columns[12]] = pd.Series(le13.fit_transform(dataset[columns[12]].astype(str)))
    dataset[columns[13]] = pd.Series(le14.fit_transform(dataset[columns[13]].astype(str)))
    dataset[columns[14]] = pd.Series(le15.fit_transform(dataset[columns[14]].astype(str)))
    dataset[columns[15]] = pd.Series(le16.fit_transform(dataset[columns[15]].astype(str)))
    dataset[columns[16]] = pd.Series(le17.fit_transform(dataset[columns[16]].astype(str)))
    print(dataset[columns[0]])
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

    cls1 = svm.SVC()
    cls1.fit(X,Y1)
    predict1 = cls1.predict(X_test1) 
    svm_acc1 = accuracy_score(y_test1,predict1)*100
    print(svm_acc1)

    cls2 = svm.SVC()
    cls2.fit(X,Y2)
    predict2 = cls2.predict(X_test2) 
    svm_acc2 = accuracy_score(y_test2,predict2)*100
    print(svm_acc2)

    cls3 = svm.SVC()
    cls3.fit(X,Y3)
    predict3 = cls3.predict(X_test3) 
    svm_acc3 = accuracy_score(y_test3,predict3)*100
    print(svm_acc3)

    cls4 = svm.SVC()
    cls4.fit(X,Y4)
    predict4 = cls4.predict(X_test4) 
    svm_acc4 = accuracy_score(y_test4,predict4)*100
    print(svm_acc4)

    cls5 = svm.SVC()
    cls5.fit(X,Y5)
    predict5 = cls5.predict(X_test5) 
    svm_acc5 = accuracy_score(y_test5,predict5)*100
    print(svm_acc5)

    cls6 = svm.SVC()
    cls6.fit(X,Y6)
    predict6 = cls6.predict(X_test6) 
    svm_acc6 = accuracy_score(y_test6,predict6)*100
    print(svm_acc6)

    cls7 = svm.SVC()
    cls7.fit(X,Y7)
    predict7 = cls7.predict(X_test7) 
    svm_acc7 = accuracy_score(y_test7,predict7)*100
    print(svm_acc7)

    cls8 = svm.SVC()
    cls8.fit(X,Y8)
    predict8 = cls8.predict(X_test8) 
    svm_acc8 = accuracy_score(y_test8,predict8)*100
    print(svm_acc8)

    cls9 = svm.SVC()
    cls9.fit(X,Y9)
    predict9 = cls9.predict(X_test9) 
    svm_acc9 = accuracy_score(y_test9,predict9)*100
    print(svm_acc9)
    cls1 = RandomForestClassifier()
    cls1.fit(X,Y1)
    cls2 = RandomForestClassifier()
    cls2.fit(X,Y2)
    cls3 = RandomForestClassifier()
    cls3.fit(X,Y3)
    cls4 = RandomForestClassifier()
    cls4.fit(X,Y4)
    cls5 = RandomForestClassifier()
    cls5.fit(X,Y5)
    cls6 = RandomForestClassifier()
    cls6.fit(X,Y6)
    cls7 = RandomForestClassifier()
    cls7.fit(X,Y7)
    cls8 = RandomForestClassifier()
    cls8.fit(X,Y8)
    cls9 = RandomForestClassifier()
    cls9.fit(X,Y9)
    
    
    color = '<font size="" color="black">'
    output = '<table border="1" align="center">'
    output+='<tr><th>SVM</th><th>Gradient Boosting</th><th>Navy Bays</th><th>DT</th><th>Random Forest</th>'
    output+='<th>SVM Output6 Accuracy</th><th>SVM Output7 Accuracy</th><th>SVM Output8 Accuracy</th><th>SVM Output9 Accuracy</th></tr>'
    output+='<tr><td>'+color+str(svm_acc1)+'</td><td>'+color+str(svm_acc2)+'</td><td>'+color+str(svm_acc3)+'</td>'
    output+='<td>'+color+str(svm_acc4)+'</td><td>'+color+str(svm_acc5)+'</td>'
    output+='<td>'+color+str(svm_acc6)+'</td><td>'+color+str(svm_acc7)+'</td>'
    output+='<td>'+color+str(svm_acc8)+'</td><td>'+color+str(svm_acc9)+'</td></tr>'
    
    output+='</table><br/><br/><br/><br/>'

    LABELS = ['No', 'Yes'] 
    conf_matrix = confusion_matrix(y_test9, predict9) 
    plt.figure(figsize =(12, 12)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("SVM Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    return render_template("ViewAccuracy.html",error=output)


@app.route('/PredictAction', methods =['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        global cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9
        global le1, le2, le3, le4, le5, le6, le7, le8
        eating = str(request.form['t1']).strip()
        activity = request.form['t2']
        bmi = request.form['t3']
        stress = request.form['t4']
        sleep = request.form['t5']
        smoking = request.form['t6']
        alcohol = request.form['t7']
        gender = request.form['t8']
        age = request.form['t9']
        data = 'Eating_Habits,Physical_Activity,BMI,Stress,Sleep,Smoking,Alcohol,Gender,Age\n'
        data+=eating+","+activity+","+bmi+","+stress+","+sleep+","+smoking+","+alcohol+","+gender+","+age

        f = open("test.csv", "w")
        f.write(data)
        f.close()
        test = pd.read_csv("test.csv")
        columns = ['Eating_Habits','Physical_Activity','BMI','Stress','Sleep','Smoking','Alcohol','Gender']
        test[columns[0]] = pd.Series(le1.transform(test[columns[0]].astype(str)))
        test[columns[1]] = pd.Series(le2.transform(test[columns[1]].astype(str)))
        test[columns[2]] = pd.Series(le3.transform(test[columns[2]].astype(str)))
        test[columns[3]] = pd.Series(le4.transform(test[columns[3]].astype(str)))
        test[columns[4]] = pd.Series(le5.transform(test[columns[4]].astype(str)))
        test[columns[5]] = pd.Series(le6.transform(test[columns[5]].astype(str)))
        test[columns[6]] = pd.Series(le7.transform(test[columns[6]].astype(str)))
        test[columns[7]] = pd.Series(le8.transform(test[columns[7]].astype(str)))
        test = test.values
        #test = normalize(test)
        print(test)
        predict1 = cls1.predict(test)
        predict2 = cls2.predict(test)
        predict3 = cls3.predict(test)
        predict4 = cls4.predict(test)
        predict5 = cls5.predict(test)
        predict6 = cls6.predict(test)
        predict7 = cls7.predict(test)
        predict8 = cls8.predict(test)
        predict9 = cls9.predict(test)
        print(str(predict1)+" "+str(predict2)+" "+str(predict3)+" "+str(predict4)+" "+str(predict5)+" "+str(predict6)+" "+str(predict7)+" "+str(predict8)+" "+str(predict9))
        msg1 = 'NO'
        msg2 = 'NO'
        msg3 = 'NO'
        msg4 = 'NO'
        msg5 = 'NO'
        msg6 = 'NO'
        msg7 = 'NO'
        msg8 = 'NO'
        msg9 = 'NO'
        if predict1[0] == 1:
            msg1 = 'YES'
        if predict2[0] == 1:
            msg2 = 'YES'
        if predict3[0] == 1:
            msg3 = 'YES'
        if predict4[0] == 1:
            msg4 = 'YES'
        if predict5[0] == 1:
            msg5 = 'YES'
        if predict6[0] == 1:
            msg6 = 'YES'
        if predict7[0] == 1:
            msg7 = 'YES'
        if predict8[0] == 1:
            msg8 = 'YES'
        if predict9[0] == 1:
            msg9 = 'YES'    

        color = '<font size="" color="black">'
        output = '<table border="1" align="center">'
        output+='<tr><th>CVD Disease</th><th>DM Disease</th><th>CKD Disease</th><th>COPD Disease</th><th>PCOD Disease</th>'
        output+='<th>HLD Disease</th><th>HTN Disease</th><th>LC Disease</th><th>AD Disease</th></tr>'
        output+='<tr><td>'+color+str(msg1)+'</td><td>'+color+str(msg2)+'</td><td>'+color+str(msg3)+'</td>'
        output+='<td>'+color+str(msg4)+'</td><td>'+color+str(msg5)+'</td><td>'+color+str(msg6)+'</td>'
        output+='<td>'+color+str(msg7)+'</td><td>'+color+str(msg8)+'</td><td>'+color+str(msg9)+'</td></tr>'
        return render_template("ViewAccuracy.html",error=output)        
                
        
@app.route("/AccuracyGraph")
def AccuracyGraph():
    height = [svm_acc1, svm_acc2, svm_acc3, svm_acc4, svm_acc5, svm_acc6, svm_acc7, svm_acc8, svm_acc9]
    bars = ('SVM Accuracy1', 'SVM Accuracy2','SVM Accuracy3','SVM Accuracy4','SVM Accuracy5','SVM Accuracy6','SVM Accuracy7','SVM Accuracy8','SVM Accuracy9')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Diseases SVM Algorithms Accuracy Graph")
    plt.show()
    plt.close()
    return render_template("AdminScreen.html")
    

@app.route("/Predict")
def Predict():
    return render_template("Predict.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/Login")
def Login():
    return render_template("Login.html")

@app.route('/UserLogin', methods =['GET', 'POST'])
def UserLogin():
    if request.method == 'POST':
        username = request.form['t1']
        password = request.form['t2']
        if username == 'admin' and password == 'admin':
            return render_template("AdminScreen.html",error='Welcome '+username)
        else:
            return render_template("Login.html",error='Invalid Login')
            

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
