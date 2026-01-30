# Simulated-Particle-Collision-Event-Analysis-with-Python
# Simulated Particle Collision Event Analysis

## Project Description
This project demonstrates a simple analysis of simulated particle collision events similar to experiments in high-energy physics.

The goal is to analyze collision data and distinguish signal events from background noise using basic data science techniques.

This project is fully reproducible using open-source tools and simulated data, making it accessible worldwide.

## Objectives
- Load particle event data
- Explore energy and momentum distributions
- Visualize detector measurements
- Perform simple event classification

## Tools Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset Description
Each row represents a collision event with:

- energy: detected particle energy (GeV)
- momentum: particle momentum
- angle: detector angle measurement
- event_type: 1 = signal event, 0 = background

## Possible Improvements
- Neural network classifier
- Use real public CERN Open Data
- Feature engineering
- Detector simulation

## Author
Student project for learning particle physics data analysis.
#Dataset (events.csv)
energy,momentum,angle,event_type
110,70,0.50,1
95,55,0.40,0
210,150,1.20,1
60,35,0.25,0
180,120,0.90,1
75,45,0.38,0
205,160,1.15,1
68,40,0.32,0
155,98,0.75,1
50,25,0.20,0
130,85,0.60,1
72,42,0.35,0
#Analysis Code (analysis.py)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("events.csv")

# Features and labels
X = data[["energy", "momentum", "angle"]]
y = data["event_type"]

# Train classifier
model = LogisticRegression()
model.fit(X, y)

# Accuracy
accuracy = model.score(X, y)
print("Model accuracy:", accuracy)

# Predictions
data["prediction"] = model.predict(X)

# Energy distribution
plt.figure()
plt.hist(data["energy"], bins=6)
plt.xlabel("Energy (GeV)")
plt.ylabel("Counts")
plt.title("Energy Distribution of Events")
plt.savefig("energy_histogram.png")
plt.show()

# Energy vs momentum plot
plt.figure()
plt.scatter(data["energy"], data["momentum"])
plt.xlabel("Energy (GeV)")
plt.ylabel("Momentum")
plt.title("Collision Events")
plt.savefig("energy_momentum_scatter.png")
plt.show()
