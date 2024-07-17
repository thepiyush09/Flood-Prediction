import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Data preparation
data = pd.read_csv('flood.csv')

df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop("FloodProbability", axis=1)
y = df["FloodProbability"]

# Train-test split (using the entire data for both training and testing since we only have one sample)
X_train, X_test, y_train, y_test = X, X, y, y

# Model selection and training
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Predicted Flood Probability: {y_pred[0]}")

# Save the model
joblib_file = "flood_prediction_model.pkl"
joblib.dump(model, joblib_file)

print(f"Model saved to {joblib_file}")

