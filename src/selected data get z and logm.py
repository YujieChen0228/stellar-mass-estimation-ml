import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from gplearn.genetic import SymbolicRegressor

perturbed_data=r'GALFORM_Vmax_perturbed_i225.csv'
df=pd.read_csv(perturbed_data)

##n(z) vs z
fig1,ax1=plt.subplots(figsize=(10,6))
ax1.hist(df['redshift'],bins=100)
ax1.set_xlabel('z',fontsize=14)
ax1.set_ylabel('n(z)',fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
plt.show()

##filter data
selected_low=df[((df['redshift']>0.24) & (df['redshift']<0.26))]
selected_medium=df[((df['redshift']>0.49) & (df['redshift']<0.51))]
selected_high=df[((df['redshift']>0.74) & (df['redshift']<0.76))]
cosmo=FlatLambdaCDM(H0=100,Om0=0.3)

d_l1=(cosmo.luminosity_distance(selected_low['redshift'])).value
M_u1=selected_low['u_app']-5*np.log10(d_l1)-25
M_g1=selected_low['g_app']-5*np.log10(d_l1)-25
M_r1=selected_low['r_app']-5*np.log10(d_l1)-25
M_i1=selected_low['i_app']-5*np.log10(d_l1)-25
M_z1=selected_low['z_app']-5*np.log10(d_l1)-25
log10m1=np.log10(selected_low['mstar_tot'])

d_l2=(cosmo.luminosity_distance(selected_medium['redshift'])).value
M_u2=selected_medium['u_app']-5*np.log10(d_l2)-25
M_g2=selected_medium['g_app']-5*np.log10(d_l2)-25
M_r2=selected_medium['r_app']-5*np.log10(d_l2)-25
M_i2=selected_medium['i_app']-5*np.log10(d_l2)-25
M_z2=selected_medium['z_app']-5*np.log10(d_l2)-25
log10m2=np.log10(selected_medium['mstar_tot'])

d_l3=(cosmo.luminosity_distance(selected_high['redshift'])).value
M_u3=selected_high['u_app']-5*np.log10(d_l3)-25
M_g3=selected_high['g_app']-5*np.log10(d_l3)-25
M_r3=selected_high['r_app']-5*np.log10(d_l3)-25
M_i3=selected_high['i_app']-5*np.log10(d_l3)-25
M_z3=selected_high['z_app']-5*np.log10(d_l3)-25
log10m3=np.log10(selected_high['mstar_tot'])

# ================= Fig2 =================
fig2, ax2 = plt.subplots(1, 5, figsize=(15,6))
bands = ['U band', 'g band', 'r band', 'i band', 'z band']
ydata2 = [M_u1, M_g1, M_r1, M_i1, M_z1]

for j in range(5):
    ax2[j].scatter(log10m1, ydata2[j], s=3)
    ax2[j].set_xlabel('log10m', fontsize=14)
    ax2[j].set_ylabel('M', fontsize=14)
    ax2[j].set_title(bands[j], fontsize=14)
    ax2[j].set_ylim(-24, -14)
    ax2[j].tick_params(axis='both', labelsize=14)

# ================= Fig3 =================
fig3, ax3 = plt.subplots(1, 5, figsize=(15,6))
ydata3 = [M_u2, M_g2, M_r2, M_i2, M_z2]

for j in range(5):
    ax3[j].scatter(log10m2, ydata3[j], s=3)
    ax3[j].set_xlabel('log10m', fontsize=14)
    ax3[j].set_ylabel('M', fontsize=14)
    ax3[j].set_title(bands[j], fontsize=14)
    ax3[j].set_ylim(-24, -14)
    ax3[j].tick_params(axis='both', labelsize=14)

# ================= Fig4 =================
fig4, ax4 = plt.subplots(1, 5, figsize=(15,6))
ydata4 = [M_u3, M_g3, M_r3, M_i3, M_z3]

for j in range(5):
    ax4[j].scatter(log10m3, ydata4[j], s=3)
    ax4[j].set_xlabel('log10m', fontsize=14)
    ax4[j].set_ylabel('M', fontsize=14)
    ax4[j].set_title(bands[j], fontsize=14)
    ax4[j].set_ylim(-24, -14)
    ax4[j].tick_params(axis='both', labelsize=14)

plt.show()
