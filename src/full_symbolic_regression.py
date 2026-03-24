from pysr import PySRRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# Prepare data
perturbed_data = r'GALFORM_Vmax_perturbed_i225.csv'
df = pd.read_csv(perturbed_data)
r1 = 0
r2 = 0.5

selected = df[((df['redshift'] > r1) & (df['redshift'] < r2))]
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
d_l = (cosmo.luminosity_distance(selected['redshift'])).value

# Calculate absolute magnitudes
M_u = selected['u_app'] - 5 * np.log10(d_l) - 25
M_g = selected['g_app'] - 5 * np.log10(d_l) - 25
M_r = selected['r_app'] - 5 * np.log10(d_l) - 25
M_i = selected['i_app'] - 5 * np.log10(d_l) - 25
M_z = selected['z_app'] - 5 * np.log10(d_l) - 25
log10m = np.log10(selected['mstar_tot'])

redshift = selected['redshift'].values
redshift2 = selected['redshift'] ** 2
redshift3 = selected['redshift'] + 1
redshift4 = 1 / (selected['redshift'] + 1)

# Calculate colors
color_ug = M_u - M_g
color_gr = M_g - M_r
color_ri = M_r - M_i
color_iz = M_i - M_z

# Define features and target
feature_names = ['color_iz', 'M_z','color_ug','color_gr','color_ri']
X = np.column_stack((color_iz.values,M_z.values,color_ug,color_gr,color_ri))
y = np.array(log10m)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up symbolic regression parameters
est_gp = PySRRegressor(
    niterations=50,
    population_size=30,
    binary_operators=["+", "-", "*", "/"],
    parsimony=0.01,
    maxsize=15,
    procs=8,  # Use 2 processes
    multithreading=True,
    verbosity=1,
    # Remove random_state to avoid warnings, accept non-determinism
    early_stop_condition=1e-5,
)

print("Starting symbolic regression model training...")
start_time = time.time()

# Train the model
est_gp.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Training completed, time: {training_time:.2f} seconds")

# Evaluate the model
print(f"Best formula: {est_gp}")
print(f"Training R²: {est_gp.score(X_train, y_train):.3f}")
print(f"Test R²: {est_gp.score(X_test, y_test):.3f}")

# Get the best formula
try:
    best_formula = est_gp.sympy()
    print(f"Best formula (SymPy): {best_formula}")
except:
    print("Unable to get SymPy formula, using string representation")

# Predict and calculate errors
y_pred = est_gp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values',fontsize=14)
plt.ylabel('Predicted Values',fontsize=14)
plt.title('Symbolic Regression: Predicted vs True Values',fontsize=14)
plt.tick_params(labelsize=14)
plt.show()


# Residual analysis function
def plot_residuals(y_true, y_pred, title_suffix=""):
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residual scatter plot
    ax1.scatter(y_true, residuals, alpha=0.5, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_xlabel('log10m (True Values)',fontsize=14)
    ax1.set_ylabel('Residuals',fontsize=14)
    ax1.tick_params(labelsize=14)
    ax1.set_title(f'Residual Analysis {title_suffix}',fontsize=14)
    ax1.set_ylim(-1.0, 1.5)

    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nR² = {r2:.4f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),fontsize=14)

    # Residual distribution histogram
    ax2.hist(residuals, bins=50, alpha=0.7, density=True)
    ax2.set_xlabel('Residuals',fontsize=14)
    ax2.set_ylabel('Density',fontsize=14)
    ax2.tick_params(labelsize=14)
    ax2.set_title('Residual Distribution',fontsize=14)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


# Plot residuals
plot_residuals(y_test, y_pred, f"(Redshift Range: {r1}-{r2})")


# Running quantile function (your original code)
def running_quantile(x, y, bins=10, quantiles=[0.25, 0.5, 0.75], min_points=5):
    """
    Efficient running quantile function (NumPy-based, mentor-friendly version)
    Parameters:
        x (array): independent variable
        y (array): dependent variable
        bins (array or int): bin edges or number of bins
        quantiles (list): list of quantiles to compute, e.g., [0.25, 0.5, 0.75]
    Returns:
        bin_centers (array): bin centers
        results (list of arrays): list of results corresponding to quantiles
    """
    if isinstance(bins, int):
        bins = np.percentile(x, np.linspace(0, 100, bins + 1))

    # Ensure bins cover the entire data range
    bins[0] = min(bins[0], np.min(x))
    bins[-1] = max(bins[-1], np.max(x))

    # Assign data to bins
    idx = np.digitize(x, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    results = []
    for q in quantiles:
        run_stat = []
        # Note: np.digitize returns indices from 1 to len(bins)
        # but index 0 represents values less than bins[0], index len(bins) represents values greater than bins[-1]
        for k in range(1, len(bins)):
            # Get data in current bin
            y_in_bin = y[idx == k]

            # If there are not enough data points in the bin, use NaN
            if len(y_in_bin) < min_points:
                run_stat.append(np.nan)
            else:
                run_stat.append(np.percentile(y_in_bin, q * 100))

        results.append(np.array(run_stat))

    return bin_centers, results


# Use running quantile function to analyze residuals
residuals = y_test - y_pred
min_val = max(1e-6, np.min(y_test[y_test > 0]))
max_val = np.max(y_test)
bins = np.logspace(np.log10(min_val), np.log10(max_val), 11)
bin_centers, [prc25, median, prc75] = running_quantile(
    y_test, residuals, bins=bins, quantiles=[0.25, 0.5, 0.75], min_points=5
)

# Plot running quantiles
plt.figure(figsize=(10, 6))
plt.plot(y_test, residuals, '.', color="cyan", alpha=0.5)
plt.plot(bin_centers, median, '-ro', fillstyle='none', markersize=10, label="median")
plt.plot(bin_centers, prc25, '--', color='blue', label="0.25")
plt.plot(bin_centers, prc75, '--', color="green", label="0.75")
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
plt.xlabel("log10m (True Values)",fontsize=14)
plt.ylabel("Residuals",fontsize=14)
plt.ylim(-1.0, 1.5)
plt.title(f" Residuals with median and percentile  {r1}-{r2}",fontsize=14)
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.show()

print("Model training and evaluation completed!")

# 简单的交叉验证（如果需要）
from sklearn.model_selection import cross_val_score


# 使用PySR的简化交叉验证
# def simple_cv_pysr(X, y, cv=3):
#     """简化的交叉验证"""
#     scores = []
#
#     for i in range(cv):
#         # 简单的手动分割
#         mask = np.random.rand(len(X)) < 0.8
#         X_train, X_test = X[mask], X[~mask]
#         y_train, y_test = y[mask], y[~mask]
#
#         model = PySRRegressor(
#             niterations=30,
#             population_size=20,
#             procs=4,  # 交叉验证时用1个进程
#             verbosity=0,
#             random_state=42
#         )
#
#         model.fit(X_train, y_train)
#         score = model.score(X_test, y_test)
#         scores.append(score)
#         print(f"CV折 {i + 1}/{cv}, R²: {score:.4f}")
#
#     return np.array(scores)
#
# # 可选：运行简化交叉验证
# cv_scores = simple_cv_pysr(X, y, cv=3)
# print(f"cross validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
