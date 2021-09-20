#PROJECT ZOMATO BENGALURU DATASET
#importing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

data=pd.read_csv('zomato.csv')
#Data Cleaning
data.drop(['url','address','phone','dish_liked','reviews_list','menu_item','listed_in(city)'],axis=1,inplace=True)
#removing duplicates
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
#cleaning rate column
def rating(val):
    if(val=='NEW' or val=='-'):
        return np.nan
    else:
        val=str(val).split('/')
        val=val[0]
        return float(val)
data['rate']=data['rate'].apply(rating)
#removing null values and replacing with 0
data.rate=data.rate.fillna(0)
#renaming approx_cost(for two people) column & Removing comma
data.rename(columns = {'approx_cost(for two people)':'approx_cost'}, inplace = True)
def remove_comma(value):
    value = str(value)
    if ',' in value:
        value = value.replace(',', '')
        return float(value)
    else:
        return float(value)
    
data['approx_cost'] = data['approx_cost'].apply(remove_comma)
data['approx_cost'].unique()
#checking unique values in rest_type column 
rest_type=data['rest_type'].value_counts()
#checking unique values in location column
data['location'].unique()
#checking unique values in cuisines column
cuisines=data['cuisines'].value_counts()

#DATA VISUALIZATION
#importing libraries
import seaborn as sns
from matplotlib import style


#Analysing relation between approx_cost(for two people) and rating
#Checking  online_order is acceptable or not
 x=data['approx_cost']
 y=data['rate']
 plt.figure(figsize=(10,7))
 sns.scatterplot(x,y,hue='online_order',data=data)
 plt.xlabel('Approx_cost')
 plt.ylabel('Rate')
 plt.xticks(rotation = 90)
 plt.show()
#conclusion:- Most of the restaurants accept online order and are in the budget.

#Calculating avg of each restaurant
 style.use('dark_background')
 data.groupby('name')['rate'].mean().nlargest(20).plot.bar()
 plt.xlabel('Name')
 plt.ylabel('Rating')
 plt.title('Avg rating')
 plt.show()

#Analysis of Top Restaurant in Banaglore
 style.use('dark_background')
 rest_name=data['name'].value_counts()[:30]
 sns.barplot(x=rest_name.index,y=rest_name)
 plt.title('Top restaurant in banaglore')
 plt.xticks(rotation=90)
 plt.show()

#Analysis on types  of Restaurants
 style.use('dark_background')
 data['rest_type'].value_counts().nlargest(40).plot.bar(color='yellow')
 plt.title('Types of Restaurants')
 plt.show()

# Popular cuisines of Bengaluru
 style.use('dark_background')
 cuisines=data['cuisines'].value_counts()[1:35]
 sns.barplot(cuisines.index,cuisines)
 plt.xlabel('cuisines')
 plt.ylabel('count')
 plt.title("Most famous cuisines of Bengaluru")
 plt.xticks(rotation=90)
 plt.show()


#Distribution of cost of two people
 style.use('ggplot')
 plt.hist(data['approx_cost'].value_counts()[:])
 plt.title('approx cost for two people')
 plt.xlabel('approx_cost')
 plt.ylabel('count')
 plt.show()


#Restaurants with table booking
style.use('ggplot')
x=data['book_table'].value_counts()
options=['not book','book']
plt.pie(x,explode=[0.0,0.2],autopct='%1.1f%%',shadow=True)
plt.title('Restaurants with table booking Option')
plt.show()

#Analysing wheather restaurant is online or not(based on price)
x=data['online_order']
y=data['approx_cost']
sns.boxplot(x,y.index,hue='online_order',data=data)
plt.xlabel('online_order')
plt.ylabel('Approx_cost')
plt.xticks(rotation = 90)
plt.show()

#Modeling
#converting strings columns into numerical columns using GetDummies(Encoding)

online_order=pd.get_dummies(data['online_order'],drop_first=True)
book_table=pd.get_dummies(data['book_table'],drop_first=True)
listed_in=pd.get_dummies(data['listed_in(type)'])
data.drop('name',axis=1,inplace=True)
data.drop(['online_order','book_table','listed_in(type)'],axis=1,inplace=True)
data=pd.concat([data,online_order,book_table,listed_in],axis=1)
# Since location, rest_type , cuisines are redundent remove them 
data.drop(['location','rest_type','cuisines'],axis=1,inplace=True)
#selection of x and y (predicting rating of restaurants)
x = data.iloc[:,1:].values
y = data.iloc[:,0:1].values

#splitting of training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=337)  

#importing linear model 
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
#checking accuracy
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
#checking MSE(Mean squared error)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))



# Testing 
y_pred1=reg.predict([[775,800,1,1,1,0,0,0,0,0,0]])
print(y_pred1)
y_pred2=reg.predict([[504,550,1,0,0,1,0,0,0,0,0]])
print(y_pred2)
y_pred3=reg.predict([[35,500,1,0,0,0,1,0,0,0,0]])
print(y_pred3)
y_pred4=reg.predict([[345,400,0,0,0,0,0,1,0,0,0]])
print(y_pred4)
y_pred5=reg.predict([[888,750,1,0,0,0,0,0,1,0,0]])
print(y_pred5)
y_pred6=reg.predict([[30,1000,0,0,0,0,0,0,0,1,0]])
print(y_pred6)
y_pred7=reg.predict([[2182,1400,1,1,0,0,0,0,0,0,1]])
print(y_pred7)
y_pred8=reg.predict([[1500,850,1,0,1,0,0,0,0,0,0]])
print(y_pred8)
y_pred9=reg.predict([[575,250,0,0,0,0,1,0,0,0,0]])
print(y_pred9)
y_pred10=reg.predict([[255,650,0,1,0,0,0,0,0,1,0]])
print(y_pred10)
y_pred11=reg.predict([[550,80,0,0,0,0,1,0,0,0,0]])
print(y_pred11)
y_pred12=reg.predict([[715,400,1,0,0,1,0,0,0,0,0]])
print(y_pred12)
y_pred13=reg.predict([[725,880,1,1,0,0,1,0,0,0,0]])
print(y_pred13)
y_pred14=reg.predict([[785,500,1,1,1,0,0,0,0,0,0]])
print(y_pred14)
y_pred15=reg.predict([[765,200,1,0,1,0,0,0,0,0,0]])
print(y_pred15)
y_pred16=reg.predict([[75,100,0,1,0,0,0,1,0,0,0]])
print(y_pred16)
y_pred17=reg.predict([[175,250,0,1,0,0,0,1,0,0,0]])
print(y_pred17)
y_pred18=reg.predict([[95,800,1,1,1,0,0,0,0,0,0]])
print(y_pred18)
y_pred19=reg.predict([[150,1000,1,0,0,0,0,1,0,0,0]])
print(y_pred19)
y_pred20=reg.predict([[120,1500,1,0,0,0,0,0,0,1,0]])
print(y_pred20)

#Comparison & Measurement Using Other Models

#DecisionTreeRegression
#importing DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
#splitting testing and training data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=303)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
#prediction
y_pred=DTree.predict(x_test)
print(y_pred)
#checking accuracy
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#RandomForestRegression
#importing RandomTreeRegression
from sklearn.ensemble import RandomForestRegressor
R_Forest=RandomForestRegressor(n_estimators=5,random_state=300,min_samples_leaf=.0001)
R_Forest.fit(x_train,y_train)
#prediction
y_pred=R_Forest.predict(x_test)
print(y_pred)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#Conclusion:Highest Accuracy Is Achieved In RandomForestRegression of 86%

#Predicting Approx Cost
x1=data.iloc[:,[0,1,3,4,5,6,7,8,9,10,11]].values
y1=data.iloc[:,2:3].values

#splitting of training and testing data
from sklearn.model_selection import train_test_split
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)  

#importing linear model 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x1_train,y1_train)
#predicting Cost
y1_pred=regressor.predict(x1_test)
print(y1_pred)
#checking accuracy
from sklearn.metrics import r2_score
print(r2_score(y1_test,y1_pred))

#DecisionTreeRegression
#importing DecisionTreeRegression
from sklearn.tree import DecisionTreeRegressor
#splitting testing and training data
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
DTree1=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree1.fit(x1_train,y1_train)
#prediction
y2_pred=DTree1.predict(x1_test)
print(y2_pred)
#checking accuracy
from sklearn.metrics import r2_score
print(r2_score(y1_test,y2_pred))

#RandomForestRegression
#importing RandomTreeRegression
from sklearn.ensemble import RandomForestRegressor
R_Forest1=RandomForestRegressor(n_estimators=5,random_state=300,min_samples_leaf=.0001)
R_Forest1.fit(x1_train,y1_train)
#prediction
y3_pred=R_Forest1.predict(x1_test)
print(y3_pred)
from sklearn.metrics import r2_score
print(r2_score(y1_test,y3_pred))
