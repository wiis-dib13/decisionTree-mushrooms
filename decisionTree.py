import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('mushrooms.csv')
print(df.shape)    #pour connaitre le nombre de ligne et colonne


x = pd.get_dummies(df[["odor", "cap-color"]])
print('color + odor', x.head())  #input color + odor


y = df["class"].map({"e": 0, "p": 1}) 
print('the class',y.head())  #output e or p

# 80% training and 20% testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(x_train,y_train)
y_prediction = model.predict(x_test)

score = accuracy_score(y_test,y_prediction)
print("accurancy",score)