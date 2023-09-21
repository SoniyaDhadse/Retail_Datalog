import pandas as pd
df=pd.read_csv("C:\\Users\\91885\\Downloads\\RetailDatalog.csv")
print(df)
remcol=['ProductID','Outlet-ID','Outlet_started_Year']
df=df.drop(remcol,axis=1)
print(df)
##remove junk characters using regular expression and list comphression
import re
df['ProductCategory']=[re.sub('\W','',i) for i in df['ProductCategory']]
print(df['ProductCategory'])
df['Outlet_Type']=[re.sub('\W','',i) for i in df['Outlet_Type']]
print(df['Outlet_Type'])
# print(df.isnull().sum())
df['ProductWeight']=df['ProductWeight'].fillna(0,inplace=False)
print(df['ProductWeight'])
df['Outlet_Size']=(df['Outlet_Size'].fillna('0',inplace=False))
print(df['Outlet_Size'])
print(df.isna().sum())
##convert values or text into numeric values i.e. preprocessing
from sklearn import preprocessing
l_encoder=preprocessing.LabelEncoder()
dfnew=df.apply(l_encoder.fit_transform)
print(dfnew)
##replace '0' into mean of column
mean=dfnew['ProductWeight'].mean()
dfnew['ProductWeight']=dfnew['ProductWeight'].replace(0,round(mean))
# print(dfnew['ProductWeight'])
mean1=dfnew['Outlet_Size'].mean()
dfnew['Outlet_Size']=dfnew['Outlet_Size'].replace(0,round(mean1))
df_reg=pd.read_csv("C:\\Users\\91885\\Downloads\\RetailDatalog_cleaned1.csv")
X=df_reg.drop(['Item_Outlet_Sales'],axis=1)
print(X)
Y=df_reg['Item_Outlet_Sales']
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
# print(X_train,X_test,Y_train,Y_test)
##step-3 Model Training
#linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
##step-4 Model fit
ycap=model.predict(X_test)
print(ycap)
df_reg_new = pd.DataFrame(Y_test)
df_reg_new['ycapvalue']=ycap
print(df_reg_new)
##step-5 Accuracy
from sklearn.metrics import accuracy_score
import numpy as np
accscore = accuracy_score(np.round(ycap),Y_test)
print(accscore*100)
##model performance tuning
##overcome underfit and overfit problem use cross validation
from sklearn.model_selection import cross_val_score
crosscore = cross_val_score(model,df_reg,Y,cv=10)
print(crosscore*100)
# df_reg['ycap']=df_reg_new['ycapvalue']
# df_reg.to_csv("C:\\Users\\91885\\Downloads\\RetailDatalog_predict.csv")



