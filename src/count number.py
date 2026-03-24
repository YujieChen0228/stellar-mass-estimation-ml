import pandas as pd


perturbed_data=r'GALFORM_Vmax_perturbed_i225.csv'
df=pd.read_csv(perturbed_data)

mask = (df['redshift'] >= 0.6 )& (df['redshift'] < 2.0)
count = df[mask].shape[0]
print(count)