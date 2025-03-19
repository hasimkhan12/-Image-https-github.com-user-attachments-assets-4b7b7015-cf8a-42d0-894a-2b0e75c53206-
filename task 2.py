import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load the dataset
file_path = r"C:\Users\vaish\OneDrive\Documents\netflix_titles.csv\netflix_titles.csv"  # Update if needed

try:
    df = pd.read_csv(r"C:\Users\vaish\OneDrive\Documents\netflix_titles.csv\netflix_titles.csv", encoding="utf-8")
except FileNotFoundError:
    print("File not found! Check the file path.")
    exit()
except PermissionError:
    print("Permission denied! Move the file to another location or run as administrator.")
    exit()

# ✅ Define the correct target column
target_column = "rating"  # You can change this to 'type', 'release_year', etc.

# ✅ Drop rows where the target is missing
df = df.dropna(subset=[target_column])

# ✅ Define features (X) and target variable (y)
X = df.drop(columns=[target_column, "title", "description", "show_id"])  # Drop text-heavy, non-useful columns
y = df[target_column]

# ✅ Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))  # Convert NaN to string before encoding
    label_encoders[col] = le

# ✅ Encode target variable (classification)
y = LabelEncoder().fit_transform(y)

# ✅ Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train a classification model (RandomForest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Make predictions
y_pred = model.predict(X_test)

# ✅ Evaluate the model
print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Feature Importance Visualization
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=model.feature_importances_)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()
