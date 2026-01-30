import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#data loading

df = pd.read_csv("car data.csv")
print(df.head())
print(df.info())

#data analysis

print(df.describe())
print(df.isnull().sum())

#data preprocessing and feature engineering
#droping car name not useful for prediction

df = df.drop("Car_Name", axis=1)

#converting year to car age

df["Car_Age"] = 2024 - df["Year"]
df = df.drop("Year", axis=1)

#encoding categorial variables
le = LabelEncoder()

df["Fuel_Type"] = le.fit_transform(df["Fuel_Type"])
df["Selling_type"] = le.fit_transform(df["Selling_type"])
df["Transmission"] = le.fit_transform(df["Transmission"])

#define feature and target

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

#train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#train regression model

model = LinearRegression()
model.fit(X_train, y_train)

#model prediction

y_pred = model.predict(X_test)

#model evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE :", rmse)
print("R2 Score :", r2)

#visualization

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()


