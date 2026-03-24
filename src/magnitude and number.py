import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###collect data
io23=r'GALFORM_Vmax_original_i23.csv'
io22=r'GALFORM_Vmax_perturbed_i225.csv'
cols_needed=['i_app']
df23=pd.read_csv(io23,usecols=cols_needed)
df22=pd.read_csv(io22,usecols=cols_needed)

###histogram
fig,ax=plt.subplots(2,1,figsize=(10,6))

ax[0].hist(df23,bins=100)
ax[0].set_yscale('log')
ax[0].set_title('when magnitude<23',fontsize=14)
ax[0].set_xlabel('magnitude',fontsize=14)
ax[0].set_ylabel("logN",fontsize=14)
ax[0].tick_params(labelsize=14)
ax[1].hist(df22,bins=100)
ax[1].hist(df22,bins=100)
ax[1].set_yscale('log')
ax[1].set_title('when magnitude<22.5',fontsize=14)
ax[1].set_xlabel('magnitude',fontsize=14)
ax[1].set_ylabel("logN",fontsize=14)
ax[1].tick_params(labelsize=14)
plt.subplots_adjust(hspace=0.4)
plt.show()

