# EX-02 Developing a Neural Network Classification Model
### Aim:
To develop a neural network classification model for the given dataset. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**DATE: 13.09.2024**
### Problem Statement:
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.
In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.
You are required to help the manager to predict the right group of the new customers.
### Neural Network Model:

Include the neural network model diagram.

### Design Steps:

- STEP 1: Import the packages and reading the dataset.
- STEP 2: Preprocessing and spliting the data.
- STEP 3: Creating a Deep Learning model with appropriate layers of depth.
- STEP 4: Plotting Training Loss, Validation Loss Vs Iteration Plot.
- STEP 5: Predicting the with Sample values.
### Program:
```Python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
df=pd.read_csv("customers.csv")
df.head()
df.info()
df.isnull().sum()
df=df.drop(['ID','Var_1'],axis=1)
df=df.dropna(axis=0)
for i in ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Segmentation']:
    print(i,":",list(df[i].unique()))
Clist=[['Healthcare','Engineer','Lawyer','Artist','Doctor','Homemaker','Entertainment','Marketing',
        'Executive'],['Male', 'Female'],['No', 'Yes'],['No', 'Yes'],['Low', 'Average', 'High']]
enc = OrdinalEncoder(categories=Clist)
df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]
    =enc.fit_transform(df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])
le = LabelEncoder()
df['Segmentation'] = le.fit_transform(df['Segmentation'])
scaler=MinMaxScaler()
df[['Age']]=scaler.fit_transform(df[['Age']])
X=df.iloc[:,:-1]
Y=df[['Segmentation']]
ohe=OneHotEncoder()
Y=ohe.fit_transform(Y).toarray()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.33,random_state=42)
model=Sequential([Dense(6,activation='relu',input_shape=[8]),Dense(10,activation='relu'),
                  Dense(10,activation='relu'),Dense(4,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy' ,metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=2000,batch_size=32,validation_data=(xtest,ytest))
metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
ypred = np.argmax(model.predict(xtest), axis=1)
ytrue = np.argmax(ytest,axis=1)
print(confusion_matrix(ytrue,ypred))
print(classification_report(ytrue,ypred))
x_single_prediction = np.argmax(model.predict(X[3:4]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```
### Output:
##### Dataset Information:
**df.head()** <br>
![image](https://github.com/user-attachments/assets/80ffc327-d8fb-40f7-8c4c-3e3fb4cf0931)

**df.info()**![image](https://github.com/user-attachments/assets/fb5f4c60-434e-4e85-9729-f4b5db091a51)
**df.isnull().sum()** <br> 
<img height=20% valign=top src="https://github.com/user-attachments/assets/1e136b13-c69d-48ec-8096-f997f14768b1">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img valign=top src="https://github.com/user-attachments/assets/220f75ea-ecce-4747-b72b-cceecbaaf431">


<table>
<tr>
<td width=50%>
  
##### Training Loss, Validation Loss Vs Iteration Plot:
![image](https://github.com/user-attachments/assets/ac534141-bee3-4abc-84db-0e54a7b6c86c)

</td> 
<td valign=top>

##### Classification Report:
![image](https://github.com/user-attachments/assets/92d06bee-96e4-416c-966b-1093f4f5ca00)

<br>
<br>

##### Confusion Matrix:
![image](https://github.com/user-attachments/assets/ec02eb64-4362-43ae-8dc9-2ad1b2510c4d)

</td>
</tr> 
</table>

##### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/0c7ad373-d03b-4ef2-82a1-fc8c6a358f89)


### Result:
A neural network classification model is developed for the given dataset.
<br>
**Developed By:VAISHALI BALAMURUGAN**<br>
**Register No :   212222230164**
