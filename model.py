#installing libraries
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#load data and check head
sexist = pd.read_csv('sexist.csv')
sexist.head()

#convert the sexist label to a numerical value
sexist['label_sexist'] = sexist['label_sexist'].map({'not sexist':0, 'sexist':1})

#set features(X) and target(y)
X = sexist['text']
y = sexist['label_sexist']

#train test split. Using 75% of the data to train and 25% to test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=22)

#Using TF-IDF Vectorizer to convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf,y_train)
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

#Check results!
print(f"Accuracy: {accuracy}")
#Accuracy: 0.736
print(classification_report(y_test,y_pred))

#               precision    recall  f1-score   support
#
#           0       0.74      0.99      0.84       362
#           1       0.71      0.07      0.13       138

#    accuracy                           0.74       500
#   macro avg       0.73      0.53      0.49       500
# weighted avg      0.73      0.74      0.65       500
