import numpy as np
import os
from flask import Flask, request, render_template, jsonify
import pickle


# import streamlit
# Create flask app
flask_app = Flask(__name__)
# Get the absolute path to the model file
script_directory = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(script_directory, "model2.pkl")
# Load the model
model = pickle.load(open(model_file_path, "rb"))


def get_age_category(age):
    if age < 25:
        return "Less than 25"
    elif age >= 25 and age <= 45:
        return "25 - 45"
    else:
        return "Greater than 45"


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predictApi", methods=["POST"])
def predictApi():
    try:
        # Get integer inputs
        age = int(request.form["age"])
        priors_count = int(request.form["priors_count"])
        v_decile_score = int(request.form["v_decile_score"])
        decile_score = int(request.form["decile_score"])
        length_of_stay = int(request.form["length_of_stay"])

        # Process categorical inputs
        c_charge_degree = request.form["c_charge_degree"]
        race = request.form["race"]
        age_cat = get_age_category(age)
        sex = request.form["sex"]

        # Map categorical inputs to encoded values based on model's expectations
        c_charge_degree_enc = {"F": 0, "M": 1}  # Example encoding, adjust as needed
        race_enc = {
            "African-American": 0,
            "Asian": 1,
            "Caucasian": 2,
            "Hispanic": 3,
            "Native American": 4,
            "Other": 5,
        }  # Example encoding, adjust as needed
        age_cat_enc = {
            "25 - 45": 0,
            "Greater than 45": 1,
            "Less than 25": 2,
        }  # Example encoding, adjust as needed
        sex_enc = {"Female": 0, "Male": 1}  # Example encoding, adjust as needed

        # Convert categorical inputs to encoded values
        c_charge_degree_enc_val = c_charge_degree_enc[c_charge_degree]
        race_enc_val = race_enc[race]
        age_cat_enc_val = age_cat_enc[age_cat]
        sex_enc_val = sex_enc[sex]

        # Prepare the input features for prediction
        features = np.array(
            [
                [
                    age,
                    priors_count,
                    v_decile_score,
                    decile_score,
                    length_of_stay,
                    c_charge_degree_enc_val,
                    race_enc_val,
                    age_cat_enc_val,
                    sex_enc_val,
                ]
            ]
        )

        # Make prediction using the model
        prediction = model.predict(features)
        # Convert prediction to a JSON serializable format
        prediction_json = prediction.tolist()
        return jsonify(prediction_json)

    except (KeyError, ValueError, IndexError) as e:
        # Handle missing or invalid inputs
        error_message = "Please provide valid input for all fields."
        return jsonify(error_message)


# @flask_app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get integer inputs
#         age = int(request.form["age"])
#         priors_count = int(request.form["priors_count"])
#         v_decile_score = int(request.form["v_decile_score"])
#         decile_score = int(request.form["decile_score"])
#         length_of_stay = int(request.form["length_of_stay"])

#         # Process categorical inputs
#         c_charge_degree = request.form["c_charge_degree"]
#         race = request.form["race"]
#         age_cat = get_age_category(age)
#         sex = request.form["sex"]

#         # Map categorical inputs to encoded values based on model's expectations
#         c_charge_degree_enc = {"F": 0, "M": 1}  # Example encoding, adjust as needed
#         race_enc = {
#             "African-American": 0,
#             "Asian": 1,
#             "Caucasian": 2,
#             "Hispanic": 3,
#             "Native American": 4,
#             "Other": 5,
#         }  # Example encoding, adjust as needed
#         age_cat_enc = {
#             "25 - 45": 0,
#             "Greater than 45": 1,
#             "Less than 25": 2,
#         }  # Example encoding, adjust as needed
#         sex_enc = {"Female": 0, "Male": 1}  # Example encoding, adjust as needed

#         # Convert categorical inputs to encoded values
#         c_charge_degree_enc_val = c_charge_degree_enc[c_charge_degree]
#         race_enc_val = race_enc[race]
#         age_cat_enc_val = age_cat_enc[age_cat]
#         sex_enc_val = sex_enc[sex]

#         # Prepare the input features for prediction
#         features = np.array(
#             [
#                 [
#                     age,
#                     priors_count,
#                     v_decile_score,
#                     decile_score,
#                     length_of_stay,
#                     c_charge_degree_enc_val,
#                     race_enc_val,
#                     age_cat_enc_val,
#                     sex_enc_val,
#                 ]
#             ]
#         )

#         # Make prediction using the model
#         prediction = model.predict(features)

#         # Format the prediction result
#         prediction_text = f"The predicted outcome is: {prediction}"

#         return render_template("index.html", prediction_text=prediction_text)

#     except (KeyError, ValueError, IndexError) as e:
#         # Handle missing or invalid inputs
#         error_message = "Please provide valid input for all fields."
#         return render_template("index.html", prediction_text=error_message)


if __name__ == "__main__":
    flask_app.run(debug=True)
