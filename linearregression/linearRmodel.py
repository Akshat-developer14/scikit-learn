# linear regression with scikit learn

from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

# 📊 Sample data: Experience vs Salary
x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])   # years of experience
y = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75])  # salary in thousands

# create a linear regression model
model = LinearRegression()
model.fit(x, y) # train the model

# Save the trained model to a file
joblib.dump(model, 'linearregression/linear_model.pkl')

# input for predicting salary
years_of_experience = float(input("Enter years of experience: "))

# predict salary
predicted = model.predict([[years_of_experience]])
print(f"Predicted salary for {years_of_experience} years of experience: ${predicted[0]:.2f}k")

# 🎯 Print weight and bias
print(f"Weight (m): {model.coef_[0]}")
print(f"Bias (c): {model.intercept_}")