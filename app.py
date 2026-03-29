from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

day_map = {
    "Monday":1,"Tuesday":2,"Wednesday":3,
    "Thursday":4,"Friday":5,"Saturday":6,"Sunday":7
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/graph")
def graph():
    return render_template("graph.html")

@app.route("/predict", methods=["POST"])
def predict():
    time = int(request.form["time"])
    day = day_map[request.form["day"]]
    vehicles = int(request.form["vehicles"])

    result = ["Low","Medium","High"][model.predict([[time,day,vehicles]])[0]]
    return result

if __name__ == "__main__":
    app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)