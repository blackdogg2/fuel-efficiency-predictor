import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower',
                'weight', 'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, names=column_names, na_values='?', sep=r'\s+', comment='\t')
data = data.dropna().drop(['car_name'], axis=1)

X = data.drop(['mpg'], axis=1)
y = data['mpg']
X = pd.get_dummies(X, columns=['origin'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    'Linear Regression': (LinearRegression(), True),
    'Ridge (alpha=1)':   (Ridge(alpha=1.0), True),
    'Lasso (alpha=0.1)': (Lasso(alpha=0.1), True),
    'ElasticNet':        (ElasticNet(alpha=0.1, l1_ratio=0.5), True),
    'Random Forest':     (RandomForestRegressor(n_estimators=200, random_state=42), False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                    max_depth=3, random_state=42), False),
    'SVR (RBF)':         (SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1), True),
}

poly_pipeline = Pipeline([
    ('poly',   PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('reg',    Ridge(alpha=1.0))
])
models['Poly Regression (deg=2)'] = (poly_pipeline, False)

results = {}

for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    Xte = X_test_scaled  if needs_scaling else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mse     = mean_squared_error(y_test, y_pred)
    rmse    = np.sqrt(mse)
    r2      = r2_score(y_test, y_pred)
    cv_in   = X_train_scaled if needs_scaling else X_train
    cv_rmse = np.sqrt(-cross_val_score(model, cv_in, y_train,
                                       cv=5, scoring='neg_mean_squared_error').mean())

    results[name] = {'model': model, 'y_pred': y_pred,
                     'MSE': mse, 'RMSE': rmse, 'R2': r2, 'CV_RMSE': cv_rmse}

    print(f"{name:<30s}  MSE={mse:7.3f}  RMSE={rmse:6.3f}  R2={r2:.4f}  CV-RMSE={cv_rmse:.3f}")


def predict_mpg(cylinders, displacement, horsepower, weight,
                acceleration, model_year, origin,
                chosen_model_name='Gradient Boosting'):
    inp = pd.DataFrame({
        'cylinders':    [cylinders],
        'displacement': [displacement],
        'horsepower':   [horsepower],
        'weight':       [weight],
        'acceleration': [acceleration],
        'model_year':   [model_year],
        'origin':       [origin]
    })
    inp = pd.get_dummies(inp, columns=['origin'], drop_first=True)
    for col in X.columns:
        if col not in inp.columns:
            inp[col] = 0
    inp = inp[X.columns]
    chosen_model, needs_scaling = models[chosen_model_name]
    inp_ready = scaler.transform(inp) if needs_scaling else inp
    return chosen_model.predict(inp_ready)[0]


print("\n-- Sample predictions --")
for mname in results:
    print(f"  {mname:<30s}: {predict_mpg(8, 307, 130, 3504, 12, 70, 1, mname):.2f} MPG")

n_models = len(results)
cols = 4
rows = (n_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx]
    ax.scatter(y_test, res['y_pred'], color='steelblue', edgecolor='k', alpha=0.6, s=40)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
    ax.set_title(f"{name}\nRMSE={res['RMSE']:.3f}  R2={res['R2']:.4f}", fontsize=9)
    ax.set_xlabel('Actual MPG')
    ax.set_ylabel('Predicted MPG')
    ax.grid(True, alpha=0.3)

for idx in range(n_models, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Actual vs Predicted MPG - All Models', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()

names    = list(results.keys())
rmses    = [results[n]['RMSE']    for n in names]
r2s      = [results[n]['R2']      for n in names]
cv_rmses = [results[n]['CV_RMSE'] for n in names]
x        = np.arange(len(names))
width    = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

bars1 = ax1.bar(x - width/2, rmses,    width, label='Test RMSE', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, cv_rmses, width, label='CV RMSE', color='tomato', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('RMSE (lower is better)')
ax1.set_title('Model Comparison - RMSE')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

ax2.bar(x, r2s, color='mediumseagreen', alpha=0.8, edgecolor='k')
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
ax2.set_ylabel('R2 Score (higher is better)')
ax2.set_title('Model Comparison - R2 Score')
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3)
for i, v in enumerate(r2s):
    ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, mname, color in [
    (ax1, 'Gradient Boosting', 'steelblue'),
    (ax2, 'Random Forest',     'tomato')
]:
    m          = results[mname]['model']
    imp        = m.feature_importances_
    idx        = np.argsort(imp)[::-1]
    feat_names = [X.columns[i] for i in idx]
    ax.bar(range(len(feat_names)), imp[idx], color=color, alpha=0.8, edgecolor='k')
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=30, ha='right')
    ax.set_title(f'{mname} - Feature Importances')
    ax.set_ylabel('Importance')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()