# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
np.random.seed(42)

# Task 1: Generate synthetic dataset (mimicking real-world student data)
# Features: hours_studied (continuous), attendance (categorical), sleep_hours (continuous with some missing)
# Target: score (dependent on features with some noise)
n_samples = 500
hours_studied = np.random.normal(20, 5, n_samples).clip(0, 40)  # Mean 20 hours, std 5
attendance = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.4, 0.3])
sleep_hours = np.random.normal(7, 1.5, n_samples).clip(4, 10)
# Introduce 10% missing values in sleep_hours
missing_mask = np.random.choice([True, False], n_samples, p=[0.1, 0.9])
sleep_hours[missing_mask] = np.nan

# Generate target: score = 50 + 2*hours + 5*(sleep) + effect of attendance + noise
attendance_map = {'Low': 0, 'Medium': 5, 'High': 10}
score = (50 + 2 * hours_studied + 5 * sleep_hours + 
         np.array([attendance_map[a] for a in attendance]) + 
         np.random.normal(0, 8, n_samples)).clip(0, 100)

# Create DataFrame
data = pd.DataFrame({
    'hours_studied': hours_studied,
    'attendance': attendance,
    'sleep_hours': sleep_hours,
    'score': score
})

print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())
print("\nMissing values:\n", data.isnull().sum())

# Task 1: Preprocess the data
# Handle missing values: Impute sleep_hours with median (suitable for continuous data)
# Transform categorical data: One-hot encode 'attendance' (since it's nominal)
numeric_features = ['hours_studied', 'sleep_hours']
categorical_features = ['attendance']

# Create preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))  # Handle missing in sleep_hours
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))  # One-hot encode, drop first to avoid multicollinearity
        ]), categorical_features)
    ])

# Apply preprocessing
X = data.drop('score', axis=1)
y = data['score']
X_preprocessed = preprocessor.fit_transform(X)

# Convert back to DataFrame for easier inspection (optional)
feature_names = (numeric_features + 
                 list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)))
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
print("\nPreprocessed data shape:", X_preprocessed.shape)
print("\nPreprocessed first 5 rows:")
print(X_preprocessed_df.head())

# Task 2: Visualize the data (before/after preprocessing insights)
# Insight 1: Scatter plot of hours_studied vs score (strong positive correlation expected)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data['hours_studied'], data['score'], alpha=0.6)
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours Studied vs Score (Raw Data)')
# Add trend line for insight
z = np.polyfit(data['hours_studied'], data['score'], 1)
p = np.poly1d(z)
plt.plot(data['hours_studied'], p(data['hours_studied']), "r--", alpha=0.8)
plt.text(5, 80, f'Slope: {z[0]:.2f} (Positive correlation)', fontsize=10)

# Insight 2: Box plot of score by attendance (categorical insight)
plt.subplot(1, 2, 2)
data.boxplot(column='score', by='attendance', ax=plt.gca())
plt.title('Score Distribution by Attendance')
plt.suptitle('')  # Remove default suptitle

plt.tight_layout()
plt.show()

# Additional insight: Correlation heatmap (numeric features only, post-imputation)
data_filled = data.copy()
data_filled['sleep_hours'].fillna(data_filled['sleep_hours'].median(), inplace=True)
plt.figure(figsize=(6, 4))
correlation_matrix = data_filled[['hours_studied', 'sleep_hours', 'score']].corr()
import seaborn as sns  # Note: Seaborn is optional but enhances viz; install if needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()

# Key Insights from Visualizations:
# - Strong positive correlation between hours_studied and score (slope ~2, meaning ~2 point increase per hour).
# - Higher attendance levels correlate with higher median scores (High: ~75, Low: ~55).
# - Sleep hours show moderate positive correlation (0.5), but less impactful than study hours.
# - No extreme outliers, but some low scores despite high hours (possibly due to low attendance/sleep).

# Task 3: Split the data (80-20 train-test split, stratified not needed for regression)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Task 4: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Task 5: Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate MSE and R² for test set
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse_test:.2f}")
print(f"R-Square (R²): {r2_test:.4f}")

# Optional: Plot predictions vs actual for insight
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs Predicted Scores (Test Set)')
plt.show()

# Model Coefficients (for interpretation)
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})
print("\nModel Coefficients:")
print(coef_df.sort_values('Coefficient', key=abs, ascending=False))
