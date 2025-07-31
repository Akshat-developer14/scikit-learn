import joblib

# Load the saved model
loaded_model = joblib.load('linearregression/linear_model.pkl')

# input for predicting salary
years_of_experience = float(input("Enter years of experience: "))

# predict salary
predicted = loaded_model.predict([[years_of_experience]])
print(f"Predicted salary for {years_of_experience} years of experience: ${predicted[0]:.2f}k")

# 🎯 Print weight and bias
print(f"Weight (m): {loaded_model.coef_[0]}")
print(f"Bias (c): {loaded_model.intercept_}")