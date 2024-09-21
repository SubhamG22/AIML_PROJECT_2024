import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv('soil_data.csv')

# Encode categorical data
soil_type_encoder = LabelEncoder()
data['soil_type'] = soil_type_encoder.fit_transform(data['soil_type'])

# Save the label encoder
joblib.dump(soil_type_encoder, 'soil_type_encoder.pkl')

# Features and labels
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'soil_type']]
y_crop = data['crop']
y_fertilizer = data['fertilizer']
y_ingredient = data['ingredient']

# Split data
X_train, X_test, y_train_crop, y_test_crop = train_test_split(X, y_crop, test_size=0.2, random_state=42)
X_train, X_test, y_train_fertilizer, y_test_fertilizer = train_test_split(X, y_fertilizer, test_size=0.2, random_state=42)
X_train, X_test, y_train_ingredient, y_test_ingredient = train_test_split(X, y_ingredient, test_size=0.2, random_state=42)

# Train models
crop_model = RandomForestClassifier()
crop_model.fit(X_train, y_train_crop)
joblib.dump(crop_model, 'crop_model.pkl')

fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(X_train, y_train_fertilizer)
joblib.dump(fertilizer_model, 'fertilizer_model.pkl')

ingredient_model = RandomForestClassifier()
ingredient_model.fit(X_train, y_train_ingredient)
joblib.dump(ingredient_model, 'ingredient_model.pkl')
print("Models have been trained and saved successfully.")
