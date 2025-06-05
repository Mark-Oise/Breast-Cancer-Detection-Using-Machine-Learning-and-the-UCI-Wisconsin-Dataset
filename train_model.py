from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing import X_scaled, y_encoded

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2,
                                                    random_state=42)


# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)