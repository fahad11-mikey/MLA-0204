from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))
print("Actual values:   ", y_test[:5])
print("Predicted values:", pred[:5])
