#train_ml.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv(r'C:/Users/Vyom Gupta/Desktop/Retrain Train AI model - Copy/data/train_dataset_reordered.csv')

# Features and target
feature_columns = ["Speed", "Weight", "Distance", "Track", "Stations"]
X = data[feature_columns]
y = data["Delay"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/train_ai_model.pkl")
print("✅ ML model saved at models/train_ai_model.pkl")
print("Accuracy:", model.score(X_test, y_test))


# ----------------------------------------
# ✅ Example Prediction Function
# ----------------------------------------

# Load model for prediction
ml_model = joblib.load("models/train_ai_model.pkl")

def predict_delays(speed, weight, distance, track, stations):
    # Ensure input is in correct format
    input_df = pd.DataFrame([[speed, weight, distance, track, stations]], columns=feature_columns)
    
    print("Input for prediction:")
    print(input_df)

    prediction = ml_model.predict(input_df)
    return prediction[0]


# 🔧 Example call (you can delete this or replace with actual values)
if __name__ == "__main__":
    delay_prediction = predict_delays(speed=50, weight=1000, distance=200, track=2, stations=5)
    print("Predicted Delay:", delay_prediction)
