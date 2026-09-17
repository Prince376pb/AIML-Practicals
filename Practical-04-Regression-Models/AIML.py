import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n = 500

area = np.random.randint(600, 3001, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)
age = np.random.randint(0, 31, n)

price = (
    area * 4500
    + bedrooms * 500000
    + bathrooms * 300000
    - age * 80000
    + np.random.normal(0, 500000, n)
)

price = np.maximum(price, 1500000)

df = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age": age,
    "Price": price.astype(int)
})

print("HOUSE PRICE DATASET")
print(df.head())
print("\nDataset Shape:", df.shape)

X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)

results = []

models = {
    "Linear Regression": linear_pred,
    "Decision Tree": tree_pred,
    "Random Forest": forest_pred
}

for name, prediction in models.items():
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)

    results.append([
        name,
        mae,
        mse,
        rmse,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"]
)

print("\nREGRESSION MODEL PERFORMANCE")
print(results_df.to_string(index=False))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

y_train_array = y_train.values

theta = np.zeros(X_train_scaled.shape[1])

learning_rate = 0.01
iterations = 1000

loss_history = []

m = len(y_train_array)

for i in range(iterations):
    predictions = X_train_scaled.dot(theta)
    error = predictions - y_train_array

    gradient = (1 / m) * X_train_scaled.T.dot(error)

    theta = theta - learning_rate * gradient

    loss = (1 / (2 * m)) * np.sum(error ** 2)
    loss_history.append(loss)

gd_predictions = X_test_scaled.dot(theta)

gd_mae = mean_absolute_error(y_test, gd_predictions)
gd_mse = mean_squared_error(y_test, gd_predictions)
gd_rmse = np.sqrt(gd_mse)
gd_r2 = r2_score(y_test, gd_predictions)

print("\nGRADIENT DESCENT LINEAR REGRESSION")
print("MAE:", round(gd_mae, 2))
print("MSE:", round(gd_mse, 2))
print("RMSE:", round(gd_rmse, 2))
print("R2 Score:", round(gd_r2, 4))

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Gradient Descent Loss Curve")
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_test, linear_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid()
plt.show()

comparison = results_df.copy()

comparison.loc[len(comparison)] = [
    "Gradient Descent",
    gd_mae,
    gd_mse,
    gd_rmse,
    gd_r2
]

print("\nFINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(comparison["Model"], comparison["R2 Score"])
plt.xlabel("Regression Model")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")
plt.xticks(rotation=20)
plt.grid(axis="y")
plt.show()

new_house = pd.DataFrame({
    "Area": [1800],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5]
})

predicted_price = linear_model.predict(new_house)

print("\nHOUSE PRICE PREDICTION")
print("Area:", new_house["Area"].iloc[0], "sq ft")
print("Bedrooms:", new_house["Bedrooms"].iloc[0])
print("Bathrooms:", new_house["Bathrooms"].iloc[0])
print("Age:", new_house["Age"].iloc[0], "years")
print("Predicted Price:", round(predicted_price[0], 2))
