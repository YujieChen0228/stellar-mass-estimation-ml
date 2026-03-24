import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

perturbed_data=r'GALFORM_Vmax_perturbed_i225.csv'
df=pd.read_csv(perturbed_data)




##collect the data




u_app=df['u_app']
g_app=df['g_app']
r_app=df['r_app']
i_app=df['i_app']
z_app=df['z_app']
redshift=df['redshift']
redshift_BCNz=df['redshift_BCNz']
mstarto=df['mstar_tot']
mstar_predicted_BCNz=df['mstar_predicted_BCNz']
u_app_perturbed=df['u_app_perturbed']
g_app_perturbed=df[ 'g_app_perturbed']
r_app_perturbed=df[ 'r_app_perturbed']
i_app_perturbed=df[ 'i_app_perturbed']
z_app_perturbed=df[ 'z_app_perturbed']
dl_BCNz=df['dL_BCNz']
u_app_corrected_BCNz=df['u_app_corrected_BCNz']
i_app_corrected_BCNz=df[ 'i_app_corrected_BCNz']
g_app_corrected_BCNz=df[ 'g_app_corrected_BCNz']
r_app_corrected_BCNz=df[ 'r_app_corrected_BCNz']
z_app_corrected_BCNz=df['z_app_perturbed']
##cosmology


cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

d_lcos_cosmo=(cosmo.luminosity_distance(redshift)).value
M_u_cos=u_app-5*np.log10(d_lcos_cosmo)-25
M_g_cos=g_app - 5 * np.log10(d_lcos_cosmo) - 25
M_r_cos=r_app - 5 * np.log10(d_lcos_cosmo) - 25
M_i_cos=i_app - 5 * np.log10(d_lcos_cosmo) - 25
M_z_cos=z_app - 5 * np.log10(d_lcos_cosmo) - 25
##BCNz




d_lcos_BCNz=(cosmo.luminosity_distance(redshift_BCNz)).value
M_g_BCNz=g_app-5*np.log10(d_lcos_BCNz)-25
M_u_BCNz=u_app - 5 * np.log10(d_lcos_BCNz) - 25
M_r_BCNz=r_app - 5 * np.log10(d_lcos_BCNz) - 25
M_i_BCNz=i_app - 5 * np.log10(d_lcos_BCNz) - 25
M_z_BCNz=z_app - 5 * np.log10(d_lcos_BCNz) - 25
#perturbed



M_g_perturbed=g_app_perturbed-5*np.log10(d_lcos_BCNz)-25
M_u_perturbed=u_app_perturbed - 5 * np.log10(d_lcos_BCNz) - 25
M_r_perturbed=r_app_perturbed - 5 * np.log10(d_lcos_BCNz) - 25
M_i_perturbed=i_app_perturbed - 5 * np.log10(d_lcos_BCNz) - 25
M_z_perturbed=z_app_perturbed- 5 * np.log10(d_lcos_BCNz) - 25
# value=df.at[3,'age']

#compute the log10(m)


log10m=np.log10(mstarto)

##compare the redshift
z_spec = redshift
z_phot = redshift_BCNz

# 统一裁剪范围，避免边缘离群点拉花
zmin, zmax = 0.0, 2.0
mask = (z_spec >= zmin) & (z_spec <= zmax) & (z_phot >= zmin) & (z_phot <= zmax)
z_spec = z_spec[mask]
z_phot = z_phot[mask]

# 计算按 z_spec 分箱的中值和16/84分位
bins = np.linspace(zmin, zmax, 50)
centers = 0.5 * (bins[1:] + bins[:-1])
p16 = np.full(centers.shape, np.nan)
p50 = np.full(centers.shape, np.nan)
p84 = np.full(centers.shape, np.nan)

Nmin = 100  # 每个bin至少多少点才画统计量，避免高z抖动
for i in range(len(bins)-1):
    m = (z_spec >= bins[i]) & (z_spec < bins[i+1])
    if np.sum(m) >= Nmin:
        vals = z_phot[m]
        p50[i] = np.median(vals)
        p16[i] = np.percentile(vals, 16)
        p84[i] = np.percentile(vals, 84)

fig, ax = plt.subplots(figsize=(10,6))

# 密度图（对数计数）
hb = ax.hexbin(z_spec, z_phot, gridsize=80, bins='log', mincnt=5)
cb = plt.colorbar(hb, ax=ax)
cb.set_label('log(count)')

# 1:1 参考线
ax.plot([zmin, zmax], [zmin, zmax], 'r--', linewidth=1.5, label='$y=x$')

# 中位线+分位带（掩掉nan）
valid = ~np.isnan(p50)
ax.plot(centers[valid], p50[valid], color='navy', lw=2, label='median')
ax.fill_between(centers[valid], p16[valid], p84[valid], color='navy', alpha=0.2, label='16–84%')

ax.set_xlim(zmin, zmax)
ax.set_ylim(zmin, zmax)
ax.set_xlabel(r'Spectroscopic redshift $z_{\rm spec}$', fontsize=14)
ax.set_ylabel(r'Photometric redshift $z_{\rm phot}$', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12, frameon=False)
plt.tight_layout()
plt.show()
fig2, axs = plt.subplots(3, 5, figsize=(15,9))
fig2.subplots_adjust(hspace=0.6, wspace=0.3)  # 调整间距

titles = [
    ["U band(cosmo)", "G band(cosmo)", "R band(cosmo)", "I band(cosmo)", "z band(cosmo)"],
    ["U band(BCNz)", "G band(BCNz)", "R band(BCNz)", "I band(BCNz)", "BCNz data in z band"],
    ["U Band(perturbed)", "G Band(perturbed)", "R Band(perturbed)", "I Band(perturbed)", "Z Band(perturbed)"]
]

y_data = [
    [M_u_cos, M_g_cos, M_r_cos, M_i_cos, M_z_cos],
    [M_u_BCNz, M_g_BCNz, M_r_BCNz, M_i_BCNz, M_z_BCNz],
    [M_u_perturbed, M_g_perturbed, M_r_perturbed, M_i_perturbed, M_z_perturbed]
]

for i in range(3):
    for j in range(5):
        axs[i,j].scatter(log10m, y_data[i][j], s=3)
        axs[i,j].set_xlabel('log10m', fontsize=14)
        axs[i,j].set_ylabel('M', fontsize=14)
        axs[i,j].set_title(titles[i][j], fontsize=14)
        axs[i,j].set_ylim(-30, 0)
        axs[i,j].tick_params(axis='both', labelsize=14)

plt.show()

