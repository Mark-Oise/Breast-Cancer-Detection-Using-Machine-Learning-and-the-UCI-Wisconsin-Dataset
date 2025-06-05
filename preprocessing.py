from data_loading import X, y
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode target labels (Malignant/Benign)
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

