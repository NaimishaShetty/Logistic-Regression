import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

data = load_digits()
digits = data['data']
labels = data['target']

plt.imshow(digits[25].reshape(8,8))
labels[25]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(digits,labels,test_size=0.2,random_state=True)

from sklearn.linear_model import LogisticRegression

logClassifier = LogisticRegression()
logClassifier.fit(x_train,y_train)

logClassifier.predict(x_test[2].reshape(1,-1))

y_test[2]

plt.imshow(x_test[2].reshape(8,8))


from sklearn.externals import joblib

joblib.dump(logClassifier,'logisticModelDigit.model')
newModel = joblib.load('logisticModelDigit.model')
newModel.predict(x_test[0].reshape(1,-1))

#from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test,y_pred)