import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('Social_Network_Ads.csv')

x=data.iloc[:,2:4].values
y=data.iloc[:,4].values

#a = data.iloc[:,2].values
#b = data.iloc[:,4].values
#plt.scatter(a,b,color='red')

from sklearn.preprocessing import StandardScaler
sScaler = StandardScaler()

x = sScaler.fit_transform(x)

#split the data

from sklearn.linear_model import LogisticRegression

lClassifier = LogisticRegression()

lClassifier.fit(x,y)

y_pred = lClassifier.predict(x)

from sklearn.metrics import accuracy_score
c=accuracy_score(y,y_pred) 

j=0
count=0
for i in y:
    if i == y_pred[j]:
        count+=1
    j+=1
print(count)
u=j-count
print(u)

u1 = input("Enter the age : ")
u2 = input("Enter the estimated salary : ")
nlist = []
nlist.append(u1)
nlist.append(u2)
nData=np.array(nlist)
nData=nData.reshape(1,-1)
sc=sScaler.transform(nData)
newPred = lClassifier.predict(sc)

print(newPred)