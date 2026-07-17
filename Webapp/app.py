from flask import Flask, render_template, request
import logging
from utils import predict_stroke
import webbrowser

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        def safe_int(val):
            return int(val)

        def safe_float(val):
            return float(val)

        input_data = [
            safe_int(request.form["gender"]),
            safe_float(request.form["age"]),
            safe_int(request.form["hypertension"]),
            safe_int(request.form["heart_disease"]),
            safe_int(request.form["ever_married"]),
            safe_int(request.form["work_type"]),
            safe_int(request.form["Residence_type"]),
            safe_float(request.form["avg_glucose_level"]),
            safe_float(request.form["bmi"]),
            safe_int(request.form["smoking_status"]),
        ]

        prediction, probability, reasons = predict_stroke(input_data)

        return render_template(
            "result.html",
            prediction=prediction,
            probability=probability,
            reasons=reasons
        )

    except Exception as e:
        logging.error(e)
        return render_template("error.html", error=str(e))


# 🔥 NEW ROUTE → GOOGLE MAPS
@app.route("/consult")
def consult():
    import webbrowser
    webbrowser.open("https://www.google.com/maps/search/neurologist+near+me")
    return "", 204


# 🔥 NEW ROUTE → STATIC DOCTORS
@app.route("/doctors")
def doctors():
    doctors_list = [
        {
            "name": "Dr. Ramesh Kumar",
            "specialization": "Neurologist",
            "hospital": "Apollo Hospital",
            "location": "Vijayawada",
            "contact": "9876543210"
        },
        {
            "name": "Dr. Priya Sharma",
            "specialization": "Neurologist",
            "hospital": "Care Hospital",
            "location": "Hyderabad",
            "contact": "9123456780"
        },
        {
            "name": "Dr. Suresh Reddy",
            "specialization": "Neurologist",
            "hospital": "AIIMS",
            "location": "Delhi",
            "contact": "9988776655"
        }
    ]

    return render_template("doctors.html", doctors=doctors_list)


if __name__ == "__main__":
    app.run(debug=True)