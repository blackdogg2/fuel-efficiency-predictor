import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load dataset because apparently suffering builds character
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

column_names = [
    'mpg', 'cylinders', 'displacement', 'horsepower',
    'weight', 'acceleration', 'model_year', 'origin', 'car_name'
]

data = pd.read_csv(
    url,
    names=column_names,
    na_values='?',
    sep=r'\s+',
    comment='\t'
)

# Clean data
data = data.dropna().drop(['car_name'], axis=1)

# Features and target
X = data.drop(['mpg'], axis=1)
y = data['mpg']

# Convert categorical origin
X = pd.get_dummies(X, columns=['origin'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models reduced to only the two survivors
models = {
    'Linear Regression': LinearRegression(),
    'SVR (RBF)': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    cv_rmse = np.sqrt(
        -cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5,
            scoring='neg_mean_squared_error'
        ).mean()
    )

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'RMSE': rmse,
        'R2': r2,
        'CV_RMSE': cv_rmse
    }

    print(f"{name:<20s} RMSE={rmse:.3f}  R2={r2:.4f}  CV_RMSE={cv_rmse:.3f}")

# Prediction function because humans love asking machines to guess things
def predict_mpg(cylinders, displacement, horsepower, weight,
                acceleration, model_year, origin,
                chosen_model='SVR (RBF)'):

    inp = pd.DataFrame({
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'model_year': [model_year],
        'origin': [origin]
    })

    inp = pd.get_dummies(inp, columns=['origin'], drop_first=True)

    for col in X.columns:
        if col not in inp.columns:
            inp[col] = 0

    inp = inp[X.columns]
    inp_scaled = scaler.transform(inp)

    return results[chosen_model]['model'].predict(inp_scaled)[0]

# Sample prediction
print("\nSample Prediction:")
print("Predicted MPG:",
      predict_mpg(8, 307, 130, 3504, 12, 70, 1))

# Plot Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(12,5))

for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(y_test, res['y_pred'], color='steelblue', edgecolor='k')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_title(f"{name}\nRMSE={res['RMSE']:.2f}, R2={res['R2']:.3f}")
    ax.set_xlabel("Actual MPG")
    ax.set_ylabel("Predicted MPG")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compare metrics
names = list(results.keys())
rmse_vals = [results[n]['RMSE'] for n in names]
r2_vals = [results[n]['R2'] for n in names]

x = np.arange(len(names))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))

ax1.bar(x, rmse_vals, color='steelblue')
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax1.set_title("RMSE Comparison")
ax1.set_ylabel("RMSE")

ax2.bar(x, r2_vals, color='mediumseagreen')
ax2.set_xticks(x)
ax2.set_xticklabels(names)
ax2.set_title("R2 Comparison")
ax2.set_ylabel("R2 Score")

plt.tight_layout()
plt.show()