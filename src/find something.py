import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,make_scorer
from sklearn.ensemble import RandomForestRegressor


perturbed_data = r'GALFORM_Vmax_perturbed_i225.csv'
df = pd.read_csv(perturbed_data)
r1=0.5
r2=2.0
selected = df[((df['redshift'] > r1) & (df['redshift'] < r2))]

cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
d_l = (cosmo.luminosity_distance(selected['redshift'])).value
M_u = selected['u_app'] - 5 * np.log10(d_l) - 25
M_g = selected['g_app'] - 5 * np.log10(d_l) - 25
M_r = selected['r_app'] - 5 * np.log10(d_l) - 25
M_i = selected['i_app'] - 5 * np.log10(d_l) - 25
M_z = selected['z_app'] - 5 * np.log10(d_l) - 25
redshift2=selected['redshift']**2
redshift3=selected['redshift']+1
log10m = np.log10(selected['mstar_tot'])
redshift = selected['redshift'].values
redshift4=1/(selected['redshift']+1)
##color
color_ug=M_u-M_g
color_gr=M_g-M_r
color_ri=M_r-M_i
color_iz=M_i-M_z


##X and Y
X=np.column_stack((
    M_i.values,
    M_z.values,
    M_r.values,
    M_g.values,
    M_u.values,
    color_ug.values,
    color_gr.values,
    color_ri.values,
    color_iz.values,
    redshift,
    redshift2.values,
    redshift3.values,
    redshift4.values
))
feature_columns = ['M_i', 'M_z', 'M_r', 'M_g', 'M_u', 'color_ug', 'color_gr', 'color_ri', 'color_iz', 'redshift','redshift2','redshift3','redshift4']
y=np.array(log10m)
##model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_residuals(y_true, y_pred, ax):
    residuals = y_true - y_pred

    # 绘制残差图
    ax.scatter(y_true, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('log10m ',fontsize=14)
    ax.set_ylabel('residuals',fontsize=14)
    ax.set_title(f' residual analysis',fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_ylim(-1.0,1.5)
    # 计算并显示性能指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nR² = {r2:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),fontsize=14)
fig,ax=plt.subplots(1,1,figsize=(8,6))

rf_model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=4)
rf_model.fit(X_train, y_train)
y_pred=rf_model.predict(X_test)
plot_residuals(y_test,y_pred,ax)

importances = rf_model.feature_importances_
normalized_importances = importances / np.sum(importances)
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': normalized_importances  # 使用归一化后的值
}).sort_values('Importance', ascending=False)
print(feature_importance_df)
# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance',fontsize=14)
plt.title(f"Feature Importances for log10M Prediction,{r1}-{r2}",fontsize=14)
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.gca().invert_yaxis() # 最重要的显示在最上面
plt.show()


def running_quantile(x, y, bins=10, quantiles=[0.25, 0.5, 0.75],min_points=5):
    """
    高效计算运行分位数的函数（基于NumPy，导师友好版）
    参数:
        x (array): 自变量
        y (array): 因变量
        bins (array or int): 分箱边界或数量
        quantiles (list): 要计算的分位数列表，如[0.25, 0.5, 0.75]
    返回:
        bin_centers (array): 箱的中心点
        results (list of arrays): 对应quantiles的计算结果列表
    """
    if isinstance(bins, int):
        bins = np.percentile(x, np.linspace(0, 100, bins + 1))

        # 确保 bins 覆盖整个数据范围
    bins[0] = min(bins[0], np.min(x))
    bins[-1] = max(bins[-1], np.max(x))

    # 将数据分配到分箱中
    idx = np.digitize(x, bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    results = []
    for q in quantiles:
        run_stat = []
        # 注意：np.digitize 返回的索引从 1 到 len(bins)
        # 但索引 0 表示小于 bins[0] 的值，索引 len(bins) 表示大于 bins[-1] 的值
        for k in range(1, len(bins)):
            # 获取当前箱中的数据
            y_in_bin = y[idx == k]

            # 如果箱中没有足够的数据点，使用 NaN
            if len(y_in_bin) < min_points:
                run_stat.append(np.nan)
            else:
                run_stat.append(np.percentile(y_in_bin, q * 100))

        results.append(np.array(run_stat))

    return bin_centers, results

# 使用函数，代码变得非常简洁和可读
residuals = y_test - y_pred
min_val = max(1e-6, np.min(y_test[y_test > 0]))
max_val = np.max(y_test)
bins = np.logspace(np.log10(min_val), np.log10(max_val), 11)
bin_centers, [prc25, median, prc75] = running_quantile(
    y_test, residuals, bins=bins, quantiles=[0.25, 0.5, 0.75], min_points=5
)

# 绘图部分完全不变
plt.plot(y_test, residuals,'.',color="cyan" )
plt.plot(bin_centers, median, '-ro', fillstyle='none', markersize=10,label="median")
plt.plot(bin_centers, prc25, '--',color='blue',label="0.25")
plt.plot(bin_centers, prc75, '--',color="green",label="0.75")
plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
plt.xlabel("log10m",fontsize=14)
plt.ylabel("residuals",fontsize=14)
plt.ylim(-1.0,1.5)
plt.title(f"residual with median and percentile,{r1}-{r2}",fontsize=14)
plt.tick_params(labelsize=14)
plt.legend(fontsize=14)
plt.show()

print(median)
print(bin_centers)
print(prc25)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(estimator=model,
#                            X=X,
#                            y=y,
#                            cv=kf, # 传入交叉验证策略
#                            scoring='neg_root_mean_squared_error', # 传入评估指标
#                            n_jobs=4, # 魔法就在这里！-1 表示使用所有CPU核心
#                            verbose=2)
# cv_scores = -cv_scores
# print(f"CV RMSE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
