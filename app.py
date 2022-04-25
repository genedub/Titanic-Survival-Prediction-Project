from flask import Flask, render_template, redirect, request
from pickle import load
import numpy as np
import pandas as pd
# Create an instance of Flask
app = Flask(__name__)

model = load(open('survival_model_trained.pkl', 'rb'))

# Route to render index.html template using data from Mongo
@app.route("/")
def home():
    ticket_class = ""
    passenger_age = ""
    ticket_price = ""
    siblings_spouses = ""
    parents_children = ""
    gender_mf = ""
    # Return template and data
    return render_template("index.html", ticketClass=ticket_class, passengerAge=passenger_age, singblingsSpouses=siblings_spouses, parentsChildren=parents_children, ticketPrice=ticket_price,  genderMF=gender_mf)

# @app.route("/submit", methods=["POST"])
# def form():
#     ticket_price = request.form["price"]
#     ticket_class = (request.form["class"])
#     passenger_age = request.form["age"]
#     siblings_spouses = request.form["sibsp"]
#     parents_children=request.form["parch"]
#     gender_mf = int(request.form["gender"])
#     predict = predict_titanic.predict(gender_mf)
#     # Right here we"ll do gender next
#     # Return template and data
    #  return render_template("index.html",ticketPrice=ticket_price, ticketClass=ticket_class, passengerAge=passenger_age, genderMF=predict)

@app.route("/overview")
def overview():
    return render_template("overview.html")

@app.route('/predict', methods=['POST'])
def predict():
    ticket_class = int(request.form["class"])
    passenger_age = float(request.form["age"])
    siblings_spouses = int(request.form["sibsp"]) 
    # talk about making this a drop down, so they cant make this a bad number 
    parents_children= int(request.form["parch"])
    ticket_price = float(request.form["price"])
    gender_mf = int(request.form["gender"])
    
    # print("ticket_class:", ticket_class)
    # print("passenger_age:", passenger_age)
    # print("siblings_spouses:", siblings_spouses)
    # print("parents_children:", parents_children)
    # print("ticket_price:", ticket_price)
    # print("gender_mf:", gender_mf)

    tt = {
           "ticket_class":ticket_class,
           "passenger_age":passenger_age,
           "siblings_spouses":siblings_spouses,
           "parents_children":parents_children,
           "ticket_price":ticket_price,
           "gender_mf":gender_mf
    }

    titanic_variable = pd.DataFrame(tt, index=[0])

    prediction_encoded = model.predict(titanic_variable)

    # print("prediction_encoded:", prediction_encoded[0])

    result = "Lived"
    if prediction_encoded[0] == 0:
        # print("ENCODED == 0 -> DIED")
        result = "Died"

    
    # We could put a list of Result Labels such as "Survivor", "Dead"
    # prediction_labels = [""]

    # features = [float(x) for x in request.form.values()]

    # final_features = [np.array(features)]
    # print(f'Data from website: {final_features}')

    # prediction_encoded = model.predict(final_features)

    # prediction = prediction_labels[prediction_encoded[0]]

    # prediction_text = f'Predicted Class:  {prediction}'
    # return render_template('index.html', prediction_text=prediction_text, features=features)
    return render_template("index.html", ticketClass=ticket_class, passengerAge=passenger_age, singblingsSpouses=siblings_spouses, parentsChildren=parents_children, ticketPrice=ticket_price, genderMF=gender_mf, result=result)

if __name__ == "__main__":
    app.run(debug=True)

# This is Lesson 3 Unit 12 Activity 10