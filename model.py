import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("traffic.csv")

df["traffic_level"] = df["traffic_level"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

X = df[["time_of_day", "day_of_week", "vehicle_count"]]
y = df["traffic_level"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")
