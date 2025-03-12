import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("calories.csv")

# Check if required columns exist
if "FoodItem" not in df.columns or "Cals_per100grams" not in df.columns:
    print("❌ ERROR: Required columns 'FoodItem' or 'Cals_per100grams' are missing in the dataset.")
    exit()

# Keep only relevant columns and drop missing values
df = df[['FoodItem', 'Cals_per100grams']].dropna()

# 🔹 Convert calorie values to numeric (remove text like 'cal')
df['Cals_per100grams'] = df['Cals_per100grams'].astype(str).str.extract('(\d+)')  # Extract numbers only
df['Cals_per100grams'] = pd.to_numeric(df['Cals_per100grams'], errors='coerce')  # Convert to float
df = df.dropna()  # Remove any NaN values after conversion

# Convert food item names into text features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['FoodItem'])

# Target Variable (Calories per 100g)
y = df['Cals_per100grams']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save Model & Vectorizer
pickle.dump(model, open("calorie_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")
