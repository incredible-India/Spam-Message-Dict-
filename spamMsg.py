from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('./spam.csv')
print(df.head())
df['target'] = df['Category'].apply(lambda x:1 if x=='spam' else 0)
print(df.head())

feature = df.Message
target = df.target


#train and test datasets
xtrain,xtest,ytrain,ytest = train_test_split(feature,target,test_size=0.2)
#conver text to vector xtrain and xtest
cv = CountVectorizer()
xtrainV = cv.fit_transform(xtrain)
xtestV = cv.transform(xtest)

#writing a model
model = MultinomialNB()

#train model
model.fit(xtrainV,ytrain)

#model score
sc=model.score(xtestV,ytest)

print(sc)

message = input("Enter the message \n")

messagev = cv.transform([message])

isspam = model.predict(messagev)

print("It is a spam message") if isspam[0] == 1 else print("It is not a spam message")
