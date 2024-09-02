from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and feature names
with open(r'C:\Users\Mohit\Desktop\ML-MINI_PROJECT\models\trained_model.pkl', 'rb') as file:
    trained_model = pickle.load(file)

# with open('feature_names.pkl', 'rb') as file:
#     feature_names = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input values from form
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    sex = int(request.form['sex'])  # 0 for Male, 1 for Female
    smoker = int(request.form['smoker'])  # 0 for Yes, 1 for No
    region = int(request.form['region'])

    # Converting gender, smoker, and region to one-hot encoded format
    gender_female = 1 if sex == 1 else 0
    gender_male = 1 if sex == 0 else 0
    smoker_no = 1 if smoker == 1 else 0
    smoker_yes = 1 if smoker == 0 else 0
    regions = [0]*4
    regions[region] = 1

    # Creating input array for prediction
    input_data = np.array([[age, bmi, children, gender_female, gender_male, smoker_no, smoker_yes] + regions])

    # Making prediction
    predicted_cost = trained_model.predict(input_data)
    print(predicted_cost)

    # Rendering prediction result back to HTML
    return render_template('index.html', prediction=predicted_cost[0])

if __name__ == '__main__':
    app.run(debug=True)
