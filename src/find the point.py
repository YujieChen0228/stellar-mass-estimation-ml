import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 读取数据
perturbed_data = 'GALFORM_Vmax_perturbed_i225.csv'
df = pd.read_csv(perturbed_data)

# 初始化列表存储结果
results = []

# 创建红移区间
redshift_bins = np.arange(0, 2.1, 0.1)
bin=0.1
# 设置宇宙学参数
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

# Bootstrapping 参数
n_bootstraps = 50  # Bootstrap 抽样次数
test_size = 0.1  # 测试集比例

# 循环处理每个红移区间
for i in range(len(redshift_bins) - 1):
    z_low = redshift_bins[i]  # 重命名为 z_low
    z_high = redshift_bins[i + 1]  # 重命名为 z_high

    # 选择当前红移区间的数据
    selected = df[(df['redshift'] > z_low) & (df['redshift'] <= z_high)]

    # 检查样本量是否足够
    if len(selected) < 1000:
        print(f"跳过区间 {z_low:.1f}-{z_high:.1f}，样本量不足: {len(selected)}")
        continue

    # 计算距离模数和绝对星等
    d_l = cosmo.luminosity_distance(selected['redshift']).value
    M_u = selected['u_app'] - 5 * np.log10(d_l) - 25
    M_g = selected['g_app'] - 5 * np.log10(d_l) - 25
    M_r = selected['r_app'] - 5 * np.log10(d_l) - 25
    M_i = selected['i_app'] - 5 * np.log10(d_l) - 25
    M_z = selected['z_app'] - 5 * np.log10(d_l) - 25

    # 计算红移相关特征
    redshift = selected['redshift'].values
    redshift2 = selected['redshift'] ** 2
    redshift3 = selected['redshift'] - 1
    redshift4=1/(selected['redshift']+1)
    # 计算颜色
    log10m = np.log10(selected['mstar_tot'])
    color_ug = M_u - M_g
    color_gr = M_g - M_r
    color_ri = M_r - M_i
    color_iz = M_i - M_z

    # 准备特征矩阵
    X = np.column_stack((
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
        redshift4
    ))

    # 特征名称
    feature_columns = ['M_i', 'M_z', 'M_r', 'M_g', 'M_u', 'color_ug', 'color_gr', 'color_ri', 'color_iz', 'redshift',
                       'redshift2', 'redshift3', 'redshift4']

    # 目标变量
    y = np.array(log10m)

    # 存储每个bootstrap的比值
    bootstrap_ratios = []
    bootstrap_r2_scores = []
    rmse_val = []
    # Bootstrapping循环
    for b in range(n_bootstraps):
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=b
        )

        # 训练随机森林模型
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=b,
            n_jobs=-1,
            max_features='sqrt'  # 添加正则化
        )
        rf_model.fit(X_train, y_train)

        # 计算特征重要性
        importances = rf_model.feature_importances_
        normalized_importances = importances / np.sum(importances)

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': normalized_importances
        })

        # 提取特定特征的重要性
        try:
            color_imp = feature_importance_df[feature_importance_df['Feature'] == 'color_iz']['Importance'].values[0]
            mz_imp = feature_importance_df[feature_importance_df['Feature'] == 'M_z']['Importance'].values[0]
            ratio = color_imp / mz_imp
            bootstrap_ratios.append(ratio)

            # 计算模型性能
            y_pred = rf_model.predict(X_test)
            r2_val = r2_score(y_test, y_pred)  # 使用不同的变量名
            bootstrap_r2_scores.append(r2_val)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            rmse_val.append(rmse)
        except (IndexError, ValueError):
            continue

    # 计算统计量
    if bootstrap_ratios:
        center = (z_low + z_high) / 2
        mean_ratio = np.mean(bootstrap_ratios)
        median_ratio = np.median(bootstrap_ratios)
        std_ratio = np.std(bootstrap_ratios)
        ci_low = np.percentile(bootstrap_ratios, 2.5)
        ci_high = np.percentile(bootstrap_ratios, 97.5)

        mean_r2 = np.mean(bootstrap_r2_scores)
        Rmse=np.mean(rmse_val)
        # 存储结果
        results.append({
            'z_low': z_low,
            'z_high': z_high,
            'z_center': center,
            'sample_size': len(selected),
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'std_ratio': std_ratio,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'mean_r2': mean_r2,
            'Rmse': Rmse
        })

        print(
            f"红移区间 {z_low:.1f}-{z_high:.1f}: 样本量={len(selected)}, 比值={mean_ratio:.3f}±{std_ratio:.3f}, R²={mean_r2:.3f}")

# 转换为DataFrame
results_df = pd.DataFrame(results)

# 绘制结果
plt.figure(figsize=(12, 8))

# 绘制均值线
plt.plot(results_df['z_center'], results_df['mean_ratio'], 'o-',
         linewidth=2, markersize=8, label='Mean Ratio')

# 绘制置信区间
plt.fill_between(
    results_df['z_center'],
    results_df['ci_low'],
    results_df['ci_high'],
    alpha=0.2,
    label='95% CI'
)

# 添加参考线
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Importance')

# 添加标签和标题
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('Ratio: color_iz / M_z', fontsize=14)
plt.title(f"Evolution of Feature Importance Ratio with Redshift\n(Bootstrapped with 95% CI),bin={bin}", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# 添加样本量标注
for i, row in results_df.iterrows():
    plt.annotate(f'{row["sample_size"] / 1000:.0f}k',
                 xy=(row['z_center'], row['mean_ratio']),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.tight_layout()
plt.savefig('feature_importance_ratio_evolution.png', dpi=300)
plt.show()

# 创建第二个图表显示模型性能
plt.figure(figsize=(10, 6))
plt.plot(results_df['z_center'], results_df['mean_r2'], 's-',
         linewidth=2, markersize=8, color='green')
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.tick_params(labelsize=14)
plt.title(f"Model Performance Across Redshift Bins,bin range={bin}", fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_performance.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results_df['z_center'], results_df['Rmse'], 's-',
         linewidth=2, markersize=8, color='green')
plt.xlabel('Redshift (z)', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.tick_params(labelsize=14)
plt.title(f"Model Performance Across Redshift Bins,bin range={bin}", fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_performance.png', dpi=300)
plt.show()
# 输出详细结果
print("\n详细结果:")
print(results_df.round(3))





