import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('BostonHousing.csv')

# Check for missing values and drop rows with missing target variable
if 'medv' in df.columns:
    df.rename(columns={'medv': 'price'}, inplace=True)
else:
    raise ValueError("The dataset does not contain the target variable 'MEDV'.")

# Define features and target variable
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax']
X = df[features]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for data preprocessing and model training
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LinearRegression())
])

# Train the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mean_ab_err = mean_absolute_error(y_test, y_pred)
mean_sq_err = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_sq_err)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mean_ab_err}")
print(f"Mean Squared Error (MSE): {mean_sq_err}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# For different models, change the 'model' step in the pipeline
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor()
}

for model_name, model in models.items():
    pipeline.set_params(model=model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mean_ab_err = mean_absolute_error(y_test, y_pred)
    mean_sq_err = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_sq_err)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel: {model_name}")
    print(f"Mean Absolute Error (MAE): {mean_ab_err}")
    print(f"Mean Squared Error (MSE): {mean_sq_err}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
