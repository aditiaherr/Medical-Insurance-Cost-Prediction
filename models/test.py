import pickle
import numpy as np

# Load the trained logistic regression model from the pickle file
with open(r'C:\Users\Mohit\Desktop\ML-MINI_PROJECT\notebooks\trained_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

# Create a sample input data array for prediction
# Age, BMI, Children, Sex (Female=0, Male=1), Smoker (No=0, Yes=1), Region (northeast=0, northwest=1, southeast=2, southwest=3)
input_data = np.array([[31, 25.74, 0, 0, 1, 0, 0, 1, 0, 0]])

# Make prediction
predicted_price = trained_model.predict(input_data)

# Print predicted price
print("Predicted insurance price:", predicted_price[0])
