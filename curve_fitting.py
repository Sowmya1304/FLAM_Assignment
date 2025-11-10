import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, dual_annealing
import matplotlib.pyplot as plt

# Loading the data
data = pd.read_csv("C:/Users/sowmy/Downloads/xy_data.csv")
x_true = data['x'].values
y_true = data['y'].values
n = len(x_true)

t_data = np.linspace(6, 60, n)

# Model 
def model_rad(t, theta_rad, M, X):
    x_pred = t * np.cos(theta_rad) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta_rad) + X
    y_pred = 42 + t * np.sin(theta_rad) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta_rad)
    return x_pred, y_pred

def L1_loss_rad(params, t, x_true, y_true):
    theta_rad, M, X = params
    x_pred, y_pred = model_rad(t, theta_rad, M, X)
    return np.sum(np.abs(x_pred - x_true) + np.abs(y_pred - y_true))

# Setting up the boundary condtions
deg_to_rad = np.pi/180.0
bounds_rad = [(0.0, 50.0 * deg_to_rad), (-0.05, 0.05), (0.0, 100.0)]

# IMPROVED: Multi-start strategy
best_loss = np.inf
best_params = None

# Try multiple initial guesses
for theta_init_deg in [10, 20, 28, 35, 45]:
    for M_init in [-0.02, 0.0, 0.02]:
        for X_init in [40, 54, 70]:
            init = [theta_init_deg * deg_to_rad, M_init, X_init]
            
            # Local optimization from this start
            res = minimize(L1_loss_rad, init, args=(t_data, x_true, y_true),
                          bounds=bounds_rad, method='L-BFGS-B',
                          options={'maxiter': 10000, 'ftol': 1e-12})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                print(f"New best! θ={res.x[0]/deg_to_rad:.4f}°, M={res.x[1]:.6f}, X={res.x[2]:.4f}, L1={res.fun:.4f}")

# Alternative: Use dual_annealing (often better than differential_evolution) 
print("\nTrying dual_annealing...")
da_result = dual_annealing(lambda p: L1_loss_rad(p, t_data, x_true, y_true),
                           bounds_rad, maxiter=1000, seed=42)
print(f"Dual annealing: θ={da_result.x[0]/deg_to_rad:.4f}°, M={da_result.x[1]:.6f}, X={da_result.x[2]:.4f}, L1={da_result.fun:.4f}")

if da_result.fun < best_loss:
    best_params = da_result.x
    best_loss = da_result.fun

# Final refinement
final_res = minimize(L1_loss_rad, best_params, args=(t_data, x_true, y_true),
                    bounds=bounds_rad, method='L-BFGS-B',
                    options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-12})

theta_rad_opt, M_opt, X_opt = final_res.x
print(f"\n=== FINAL RESULT ===")
print(f"θ = {theta_rad_opt/deg_to_rad:.6f}° ({theta_rad_opt:.6f} rad)")
print(f"M = {M_opt:.8f}")
print(f"X = {X_opt:.6f}")
print(f"L1 loss = {final_res.fun:.6f}")

# --- Plot ---
x_fit, y_fit = model_rad(t_data, theta_rad_opt, M_opt, X_opt)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: x-y space
axes[0].plot(x_true, y_true, 'bo', markersize=4, label='Actual', alpha=0.7)
axes[0].plot(x_fit, y_fit, 'g-', linewidth=2, label='Fit')
axes[0].legend()
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f'Fit Quality: L1={final_res.fun:.2f}')
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
axes[1].plot(t_data, x_true - x_fit, 'b-', label='x residuals')
axes[1].plot(t_data, y_true - y_fit, 'r-', label='y residuals')
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].legend()
axes[1].set_xlabel('t')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residual Analysis')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print residual statistics
x_res = x_true - x_fit
y_res = y_true - y_fit
print(f"\nResidual stats:")
print(f"X: mean={np.mean(np.abs(x_res)):.4f}, max={np.max(np.abs(x_res)):.4f}")
print(f"Y: mean={np.mean(np.abs(y_res)):.4f}, max={np.max(np.abs(y_res)):.4f}")