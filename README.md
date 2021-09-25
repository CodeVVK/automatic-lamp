# automatic-lamp
Using linear regression predict the total sales based on amount spend on marketing. 
import pandas as pd
marketing = pd.read_csv ("/content/drive/MyDrive/ML/ML_Assign.csv")
print (marketing)
test_data = marketing[['expense']].values
test_sale = marketing[['sales']].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(test_data,test_sale, test_size = 0.4, random_state = 42 )
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
lr = LinearRegression()
lr.fit(x_train,y_train)#learning phase
x1 = lr.predict(x_test)#Making predictions to test the model on test data
print ('\n Array: \n',x1)
x2 = lr.score(x_test,y_test)#Test accuracy
x3 = lr.score(x_train,y_train)#Train accuracy
print ('\n Train accuracy Score: \n',x3)
print ('\n Test accuracy Score: \n',x2)

