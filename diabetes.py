import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



diabetes_data = pd.read_csv('diabetes.csv')  #include the diabetes dataset to a pandas dataframs

"""
    print(diabetes_data.head()) #print first 5 rows
    diabetes_data.shape   #print dimension
    print(diabetes_data.describe())  #statistical measures of the data(min, max, mean, etc.,)
    print(diabetes_data['Outcome'].value_counts())  #count of frequency
    print(diabetes_data.groupby("Outcome").mean())   #Group the data and find the mean 

"""

x = diabetes_data.drop(columns="Outcome", axis=1) #seperate data and label, axis 1 => col axis 0=> row
y = diabetes_data["Outcome"]

#Data standardization

scalar = StandardScaler()
scalar.fit(x)
stand_data = scalar.transform(x) #or scalar.fit() both function are same

#Train Test Split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y, random_state=2)

#Training the Model

classifier = svm.SVC(kernel='linear')

#Training the support vector machine classifier

classifier.fit(x_train, y_train)

#Model Evaluation

x_train_prediction = classifier.predict(x_train)
training_data_accurancy = accuracy_score(x_train_prediction, y_train) #print("Accurancy Score of the training data : ", training_data_accurancy)

x_test_prediction = classifier.predict(x_test)
test_data_accurancy = accuracy_score(x_test_prediction, y_test)  #print("Accurancy Score of the test data = ", test_data_accurancy)

#Making Predictive System

pregnancies = float(input("Enter a Pregnancies value = "))
Glucose = float(input("Enter a Glucose value = "))
bloodpressure = float(input("Enter a BloodPressure value = "))
skinthickness = float(input("Enter a skinthickness value = "))
insulin = float(input("Enter a insulin value = "))
bmi = float(input("Enter a BMI value = "))
diabetespedigreefunction = float(input("Enter a diabetespedigreefunction value = "))
age = float(input("Enter a age value = "))

input_data = [pregnancies] + [Glucose] + [bloodpressure] + [skinthickness] + [insulin] + [bmi] + [diabetespedigreefunction] + [age]

np_array = np.asarray(input_data)

#reshape the array as we are predicting for on instance
input_data_reshaped = np_array.reshape(1, -1)

input_standard = scalar.transform(input_data_reshaped)

prediction = classifier.predict(input_standard)

if(prediction[0] == 0):
    print("THe presion is not diabetic")
else:
    print("The persion is Diabetic")

