import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv("traindata.csv")
df.head(5)
#print(df.head(5))
df.shape
df.isnull().sum()
#print(df.isnull().sum())

#removing seats
df['Seats'].mode()
df['Seats'].fillna(value=5.0,inplace=True)
df.info()

#removing kmpl and km/kg from mileage column
df['Mileage'] = df['Mileage'].apply(lambda x: str(x).replace('kmpl', '') if 'kmpl' in str(x) else str(x))
df['Mileage'] = df['Mileage'].apply(lambda x:str(x).replace('km/kg', '') if 'km/kg' in str(x) else str(x))

#removing CC from engine column
df['Engine'] = df['Engine'].apply(lambda x: str(x).replace('CC', '') if 'CC' in str(x) else str(x))

#removing bhp from power column
df['Power'] = df['Power'].apply(lambda x: str(x).replace('bhp', '') if 'bhp' in str(x) else str(x))

df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Engine'] = pd.to_numeric(df['Engine'], errors='coerce')
df['Power'] = pd.to_numeric(df['Power'], errors='coerce')

df['Mileage'].mode()
df['Mileage'].fillna(value=17.0,inplace=True)
df['Engine'].mode()
df['Engine'].fillna(value=1197.0,inplace=True)
df['Power'].mode
df['Power'].fillna(value=74,inplace=True)

df.isnull().sum()
#print(df.isnull().sum())
#Since new price column has so many values will not use


df['Name'].nunique()
#print(df['Name'].nunique())
#The Name column has so many values so we will separate the brand names from the column and create a new column Brand_Name.

df['Brand_Name'] = df['Name'].str.split(' ').str[0]
df.groupby('Brand_Name').nunique()
df['Brand_Name'].unique()
#print(df['Brand_Name'].unique())

#dropping the Name ,Location and new_price column
df1_map=df.drop(["Name","Location","New_Price"],axis='columns')
df1_map.head(5)
#print(df1_map.head(5))
#new data frame

#heat map to find no null values in dataset
#sns.heatmap(df1_map.isnull())
#plt.show()

plt.xlabel("Brand_Name")
plt.ylabel("Count of car")
#df1_map['Brand_Name'].value_counts().plot(kind='bar',title='Brand vs Car count',color='#C03928')
#plt.grid(color='black',linestyle='-.',linewidth=0.7)        
#plt.show()
#The abv graph shows people buy Maruti and Hyundai car more than other brands

plt.xlabel("Year")
plt.ylabel("Count of car")
#df1_map['Year'].value_counts().plot(kind='bar',title='Year vs car count',color='#0E6251')
#plt.grid(color='black', linestyle='-.', linewidth=0.7)
#plt.show()
#Abv graph shows maximum number of cars in the data frame is between 2010 to  2017

#fuel-type
plt.xlabel("Fuel_Type")
plt.ylabel("Count of car")
#df1_map['Fuel_Type'].value_counts().plot(kind='bar',title='Fuel_Type vs car count',color='black')
#plt.grid(linestyle='-.')
#plt.show()

#Transmission
plt.xlabel("Transmission")
plt.ylabel("Count of car")
#df1_map['Transmission'].value_counts().plot(kind='bar',title='Transmission vs car count',color='#C0392B')
#plt.grid(linestyle='-.')
#plt.show()

#owner type
plt.xlabel("Owner_Type")
plt.ylabel("Count of car")
#df1_map['Owner_Type'].value_counts().plot(kind='bar',title='Owner_Type vs car count',color='blue')
#plt.grid(linestyle='-.')
#plt.show()

#seats
plt.xlabel("No of seats")
plt.ylabel("Count of car")
#df1_map['Seats'].value_counts().plot(kind='bar',title='Number of seats vs car count',color='cyan')
#plt.grid(linestyle='-.')
#plt.show()

#CONCLUSION 1.Max cars are of petrol  2.Manual cars are more  3.First hand cars are max  4.Cars with 5 seater are dominant

BrandVsPrice=pd.DataFrame(df1_map.groupby('Brand_Name')['Price'].mean())
BrandVsPrice.plot.bar(color='tomato',figsize=(11,5))
#plt.grid(linestyle='-.')
#plt.show()

#abv graph show Lamborghini is the most expensive car

#year vs price
plt.title("Year vs Price")
plt.xlabel("Year")
plt.ylabel("Price")
#plt.scatter(df1_map.Year,df1_map.Price)
#plt.show()

#fuel type vs price
plt.title("Fuel_Type vs Price")
plt.xlabel("Fuel_Type")
plt.ylabel("Price")
#plt.scatter(df1_map.Fuel_Type,df1_map.Price)
#plt.show()

#transmission vs price
plt.title("Transmission vs Price")
plt.xlabel("Transmission")
plt.ylabel("Price")
#plt.scatter(df1_map.Transmission,df1_map.Price)
#plt.show()

#owner type vs price
plt.title("Owner_Type vs Price")
plt.xlabel("Owner")
plt.ylabel("Price")
#plt.scatter(df1_map.Owner_Type,df1_map.Price)
#plt.show()


#Cars ranging between the years 2012 to 2020 cost more.
#Petrol and diesel cars are costly.
#Automatic cars cost more 
#First hand cars are costly

plt.title("Kilometers Driven vs Price")
plt.xlabel("Kilometers Driven")
plt.ylabel("Price")
#plt.scatter(df1_map.Kilometers_Driven,df1_map.Price)
#plt.show()

#one of the cars has km drove more than 6500000, this is an outliner and we need to remove
#removing outlier
df1_map.drop(df1_map[df1_map['Kilometers_Driven'] >= 6500000].index, axis=0, inplace=True)

#mileage vs price
plt.title("Mileage vs Price")
plt.xlabel("Mileage")
plt.ylabel("Price")
#plt.scatter(df1_map.Mileage,df1_map.Price)
#plt.show()

#Seats vs price
plt.title("Seats vs Price")
plt.xlabel("Seats")
plt.ylabel("Price")
#plt.scatter(df1_map.Seats,df1_map.Price)
#plt.show()
#Some rows have zero values in mileage and seats column

df1_map.isin([0]).sum()
#print(df1_map.isin([0]).sum())

#Dropping 1 row from Seats column with zero value
df1_map.drop(df1_map[df1_map['Seats']==0].index,axis=0,inplace=True)

#Cant drop rows with zero value in the ,ileage column sice we lose 68 rows, so we replace with dummy values
#we have already calculated the mode of milage column for filling
#null values which is 17.0
df1_map["Mileage"].replace({0.0:17.0 },inplace=True)

df1_map.isin([0]).sum()
#print(df1_map.isin([0]).sum())

#Machine Learning algorithms work with a numeric value.
#creating a new dataframe 
df2_n = df1_map.copy()
#Fuel type, Transmission, Owner type,and Brand Name are categorical columns

from sklearn.preprocessing import LabelEncoder
le_Fuel_Type=LabelEncoder()
le_Transmission=LabelEncoder()
le_Owner_Type=LabelEncoder()
le_Brand_Name=LabelEncoder()
df2_n['Fuel_Type_n']= le_Fuel_Type.fit_transform(df2_n['Fuel_Type'])
df2_n['Transmission_n']=le_Transmission.fit_transform(df2_n['Transmission'])
df2_n['Owner_Type_n']=le_Owner_Type.fit_transform(df2_n['Owner_Type'])
df2_n['Brand_Name_n']=le_Brand_Name.fit_transform(df2_n['Brand_Name'])

df2_n.head(1)
#print(df2_n.head(5))

#4 columns are created with numeric values

#Dropping columns with data type object
df2_n=df2_n.drop(["Fuel_Type","Transmission","Owner_Type","Brand_Name"],axis='columns')
df2_n.head(5)
#print(df2_n.head(5))

#Shuffling columns as per our needs
df2_n=df2_n[['Brand_Name_n','Year','Kilometers_Driven','Fuel_Type_n','Transmission_n','Owner_Type_n','Mileage','Engine','Power','Seats','Price']]
df2_n.head(1)
#print(df2_n.head(1))

#Correlation matrix
corrMatrix = df2_n.corr()
plt.figure(figsize=(10,7))
#sns.heatmap(corrMatrix, annot=True,cmap= 'coolwarm', linewidths=3, linecolor='black')
#plt.show()

#Creating 2 new data frames
df3_inputs=df2_n.drop(["Price"],axis='columns')
df3_target=df2_n['Price']
df3_inputs.head(5)
#print(df3_inputs.head(5))
df3_target.head(5)
#print(df3_target.head(5))

#‘df3_inputs’ data frame has input features and ‘df3_target’ data frame has the target value that we need to predict i.e price.

#least and imp feature used for predictiion
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(df3_inputs,df3_target)
#use inbuilt class feature_importances of ExtraTreeRegressor
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=df3_inputs.columns)
plt.figure(figsize=(11,5))
plt.xlabel("Value")
plt.ylabel("Features")
plt.title("Features vs Importance")
plt.grid()
#feat_importances.nlargest(10).plot(kind='barh',color='#D98880')##45B39D
#plt.grid(color='black', linestyle='-.', linewidth=0.7)
#plt.show()


#Transmission_n and Owner_Type_n are the most and least important features for predicting the price of a used car.

#Applying different modes on the data

from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#Splitting data as test and rain
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df3_inputs,df3_target,test_size=0.2,random_state=10)
len(X_train)
#print(len(X_train))
len(X_test)
#print(X_test)

#Training
Model_RandomForest = RandomForestRegressor(max_features='sqrt',bootstrap='True')
Model_RandomForest.fit(X_train,y_train)
RandomForestRegressor(bootstrap='True', ccp_alpha=0.0,criterion='mse',
                      max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2,min_weight_fraction_leaf=0.0,
                      n_estimators=100, n_jobs=None, oob_score=False,
                      random_state=None,verbose=0, warm_start=False)
Model_RandomForest.score(X_test,y_test)
#print(Model_RandomForest.score(X_test,y_test))

#importing model
#pickel method
import pickle

#wrinting the model in a file
pickle.dump(Model_RandomForest,open('rmodel.pk1','wb'))

#reading the file
rmodel=pickle.load(open('rmodel.pk1','rb'))

#test model
rmodel.predict([[1,2016,10000,15,2000,300,5.0,3,1,0]])
print(rmodel.predict([[1,2016,10000,15,2000,3000,5.0,3,1,0]]))
