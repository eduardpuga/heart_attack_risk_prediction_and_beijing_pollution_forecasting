import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data/heart.csv')

# Separate features and target
X = data.drop(columns=['output'])
y = data['output']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the scaler on the training data and save it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = RandomForestClassifier(max_depth=None, min_samples_split=10, n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the fitted scaler
with open('heart_attack/models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the trained model
with open('heart_attack/models/best_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model and scaler saved successfully.")