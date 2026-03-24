import numpy as np
import matplotlib.pyplot as plt

# --- load data ---
file = r'zdl.dat'
data = np.loadtxt(file, comments='#')
z = data[:, 0]
dl = data[:, 1]

# --- restrict to z < 2 for fitting, as advisor suggested ---
mask = z < 2.0
z_fit = z[mask]
dl_fit = dl[mask]

# optional: sort by z for nicer plotting lines
order = np.argsort(z)
z = z[order]
dl = dl[order]
order_fit = np.argsort(z_fit)
z_fit = z_fit[order_fit]
dl_fit = dl_fit[order_fit]

# --- linear fit on the restricted range ---
# use numpy's robust polyfit instead of manual sums
k, b = np.polyfit(z_fit, dl_fit, 1)

# line for plotting (only across the fitted range)
y_fit_line = k * z_fit + b

# --- plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# labels & font sizes (bigger, to match body text)
ax.set_xlabel('Redshift z', fontsize=14)
ax.set_ylabel('Luminosity distance (Mpc/h)', fontsize=14)

# raw curve/points (whatever zdl.dat represents)
ax.plot(z_fit, dl_fit, 'k.', markersize=2, label='data (z<2)')
ax.plot(z_fit, y_fit_line, 'r--', label=f'linear fit (z<2)')
ax.set_xlim(0, 2)
ax.set_ylim(0, dl_fit.max()*1.1)  # y 轴上限设为数据最大值的 1.1 倍

# axis/tick formatting           # focus on the relevant range
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()

