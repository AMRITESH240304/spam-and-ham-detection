# creating ML model to predict Spam and ham (SPAM) or (SMS)
import pandas as pd 
import numpy as np 
import matplotlib as plt 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

mail_sms = pd.read_csv('D:\spam and ham\spam1.csv')
# print(mail_sms.shape)-->(5572, 5)
mail_data = mail_sms.where((pd.notnull(mail_sms)),'')
# label spam as '0' && ham as '1'
mail_data.loc[mail_data['v1'] == 'spam', 'v1',] = 0
mail_data.loc[mail_data['v1'] == 'ham', 'v1',] = 1

X = mail_data['v2']
y = mail_data['v1']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=3)

# Transform the text data to feature vectors that can be used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_feature = feature_extraction.fit_transform(X_train)
X_test_feature = feature_extraction.transform(X_test)

# Convert y_train and y_test to integer data type
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Create an instance of the SVM model and train it on the training set
model = SVC()
model.fit(X_train_feature, y_train)

# Evaluate the performance of the model on the testing set
accuracy = model.score(X_test_feature, y_test)
#print('Accuracy:', accuracy*100,'%') -> (98.29596412556054 %)

mail = input("type mail:: ")
input_mail = feature_extraction.transform([mail])
prediction = model.predict(input_mail)

if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')