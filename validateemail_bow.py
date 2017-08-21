
# coding: utf-8

# In[82]:

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn.metrics import ( precision_score, recall_score,
                             f1_score, accuracy_score)
from sklearn.model_selection import train_test_split

dic = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
      '0','1','2','3','4','5','6','7','8','9','@']

csvpath = input('Enter the complete path of csv file :  ')
data = np.zeros((1,37), dtype=np.int)
target = np.zeros((1,), dtype=np.int)

print("Extracting features from raw data..")
with open(csvpath,'r') as csvfile:
    emaildata = csv.reader(csvfile)
    i=0
    for email in emaildata:
        if i!=0:
            temp = np.zeros((1,37), dtype=np.int)
            data = np.append(data,temp,axis=0)
            temp = np.zeros((1,), dtype=np.int)
            target = np.append(target,temp,axis=0)
        
        if email[1] == "0":
            target[i]=1
        
        email = email[0].lower()
        for c in email:
            if c==".":
                break
            id = dic.index(c)
            data[i,id] += 1
            
        #print(data[i,])
        #print(target[i,])
        i += 1
       
        
     
    print("Spliting dataset: 75% training and 25% testing\n")
    X_train, X_test, y_train, y_test = train_test_split(data,target)
    
   # print (y_test)

    pca = decomposition.PCA()
    pca.fit(X_train)
    
    plt.figure(1)
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance')
    plt.show()

    logreg = linear_model.LogisticRegression()

    print("Training logistic regression classifier..")
    logreg.fit(X_train, y_train)

    print("Testing logistic regression classifier..")
    y_pred = logreg.predict(X_test)

    acc = accuracy_score(y_test, y_pred)*100
    print("Accuracy : ", acc,"%" )
    print("Precision: %1.3f" % precision_score(y_test, y_pred))
    print("Recall: %1.3f" % recall_score(y_test, y_pred))
    print("F1: %1.3f\n" % f1_score(y_test, y_pred))
    


# In[ ]:



