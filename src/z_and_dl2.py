import numpy as np
import matplotlib.pyplot as plt

# --- load data ---
file = r'zdl.dat'
data = np.loadtxt(file, comments='#')
z = data[:, 0]
dl = data[:, 1]

# --- restrict to z < 2 for fitting/plotting ---
mask = z < 2.0
z_fit = z[mask]
dl_fit = dl[mask]

# sort for nicer lines
order = np.argsort(z_fit)
z_fit = z_fit[order]
dl_fit = dl_fit[order]

# --- models: linear & quadratic ---
# linear: dl ~ k*z + b
k_lin, b_lin = np.polyfit(z_fit, dl_fit, 1)
y_lin = k_lin * z_fit + b_lin

# quadratic: dl ~ a*z^2 + b*z + c
a_quad, b_quad, c_quad = np.polyfit(z_fit, dl_fit, 2)
y_quad = a_quad * z_fit**2 + b_quad * z_fit + c_quad

# --- simple metrics (RMSE) ---
rmse_lin  = np.sqrt(np.mean((dl_fit - y_lin)**2))
rmse_quad = np.sqrt(np.mean((dl_fit - y_quad)**2))
print(f"Linear RMSE   : {rmse_lin:.2f} Mpc/h")
print(f"Quadratic RMSE: {rmse_quad:.2f} Mpc/h")

# --- plot ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xlabel('Redshift z', fontsize=14)
ax.set_ylabel('Luminosity distance [Mpc/h]', fontsize=14)

# data
ax.plot(z_fit, dl_fit, 'k.', markersize=2, label='data (z<2)')

# fits
ax.plot(z_fit, y_lin,  'r--', linewidth=2,
        label=f'linear fit (z<2): y={k_lin:.2f} z + {b_lin:.2f}')
ax.plot(z_fit, y_quad, 'b-',  linewidth=2,
        label=f'quadratic fit (z<2): y={a_quad:.2f} z² + {b_quad:.2f} z + {c_quad:.2f}')

ax.set_xlim(0, 2)
ax.set_ylim(0, dl_fit.max()*1.05)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12, frameon=False)
plt.tight_layout()
plt.show()
