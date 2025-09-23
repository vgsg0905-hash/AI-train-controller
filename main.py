#main.py
import pandas as pd
import os

df = pd.read_csv(r'C:/Users/Vyom Gupta/Desktop/Train AI model/data/train_data.csv')
print(df)

print(os.getcwd())  # shows where Python/Streamlit is running
print(os.path.exists("models/ppo_multitrain.zip"))

