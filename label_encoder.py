#label_encoder.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv(r'C:/Users/Vyom Gupta/Desktop/Retrain Train AI model - Copy/data/train_dataset_reordered.csv')

# Encode Signal column
if data["Signal"].dtype == object:
    le_signal = LabelEncoder()
    data["Signal"] = le_signal.fit_transform(data["Signal"])
    joblib.dump(le_signal, "models/signal_encoder.pkl")
    print("✅ Encoded Signal column:", dict(zip(le_signal.classes_, le_signal.transform(le_signal.classes_))))
else:
    print("Signal column is already numeric.")

# Encode Track column
if data["Track"].dtype == object:
    le_track = LabelEncoder()
    data["Track"] = le_track.fit_transform(data["Track"])
    joblib.dump(le_track, "models/track_encoder.pkl")
    print("✅ Encoded Track column:", dict(zip(le_track.classes_, le_track.transform(le_track.classes_))))
else:
    print("Track column is already numeric.")

# Convert Stations column to float
if data["Stations"].dtype != float:
    data["Stations"] = data["Stations"].astype(float)
    print("✅ Converted Stations column to float.")

# Save preprocessed dataset
data.to_csv(r'C:/Users/Vyom Gupta/Desktop/Retrain Train AI model - Copy/data/train_dataset_reordered.csv', index=False)
print("💾 Preprocessed dataset saved as train_data_processed.csv")
