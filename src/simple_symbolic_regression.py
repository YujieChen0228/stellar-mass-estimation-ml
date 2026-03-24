from gplearn.genetic import SymbolicRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# 加载数据
perturbed_data = r'E:\GALFORM_Vmax_files\GALFORM_Vmax_perturbed_i225.csv'
df = pd.read_csv(perturbed_data)
selected = df[((df['redshift'] > 0.49) & (df['redshift'] < 0.51))]

# 计算绝对星等
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
d_l = (cosmo.luminosity_distance(selected['redshift'])).value
M_u = selected['u_app'] - 5 * np.log10(d_l) - 25
M_g = selected['g_app'] - 5 * np.log10(d_l) - 25
M_r = selected['r_app'] - 5 * np.log10(d_l) - 25
M_i = selected['i_app'] - 5 * np.log10(d_l) - 25
M_z = selected['z_app'] - 5 * np.log10(d_l) - 25
log10m = np.log10(selected['mstar_tot'])

# 计算颜色指数
color_gr = M_g - M_r

# 准备数据
redshift = selected['redshift'].values

# i_band
X_phase1 = M_i.values.reshape(-1, 1)
y = np.array(log10m)

# i band with g-r color
X_phase2 = np.column_stack((M_i.values, color_gr.values))

# i_band, g-r, z
X_phase3 = np.column_stack((M_i.values, color_gr.values, redshift))


# 简化的残差分析函数
def plot_residuals(y_true, y_pred, phase_name, ax):
    residuals = y_true - y_pred

    # 绘制残差图
    ax.scatter(y_true, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel('log10m ')
    ax.set_ylabel('residuals')
    ax.set_title(f'{phase_name} residual analysis')

    # 计算并显示性能指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'RMSE = {rmse:.4f}\nR² = {r2:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# 主程序
if __name__ == "__main__":
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('residuals', fontsize=16)

    # 阶段一
    X_train, X_test, y_train, y_test = train_test_split(X_phase1, y, test_size=0.2, random_state=42)
    model_phase1 = SymbolicRegressor(
        population_size=2000,
        generations=50,
        function_set=['add', 'sub', 'mul'],
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        parsimony_coefficient=0.05,
        random_state=42,
        verbose=1
    )
    model_phase1.fit(X_train, y_train)
    y_pred_phase1 = model_phase1.predict(X_test)
    plot_residuals(y_test, y_pred_phase1, "first", axes[0])
    print(f"first output: {model_phase1._program}")

    # 阶段二
    X_train, X_test, y_train, y_test = train_test_split(X_phase2, y, test_size=0.2, random_state=42)
    model_phase2 = SymbolicRegressor(
        population_size=2000,
        generations=50,
        function_set=['add', 'sub', 'mul'],
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        parsimony_coefficient=0.05,
        random_state=42,
        verbose=1
    )
    model_phase2.fit(X_train, y_train)
    y_pred_phase2 = model_phase2.predict(X_test)
    plot_residuals(y_test, y_pred_phase2, "second", axes[1])
    print(f"second output: {model_phase2._program}")

    # 阶段三
    X_train, X_test, y_train, y_test = train_test_split(X_phase3, y, test_size=0.2, random_state=42)
    model_phase3 = SymbolicRegressor(
        population_size=2000,
        generations=50,
        function_set=['add', 'sub', 'mul'],
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        parsimony_coefficient=0.05,
        random_state=42,
        verbose=1
    )
    model_phase3.fit(X_train, y_train)
    y_pred_phase3 = model_phase3.predict(X_test)
    plot_residuals(y_test, y_pred_phase3, "third", axes[2])
    print(f"the third output: {model_phase3._program}")

    plt.tight_layout()
    plt.savefig('residuals_comparison.png', dpi=300)
    plt.show()

    # 交叉验证
    print("\n=== crossvalidation ===")
    for phase_name, X_phase in [("first", X_phase1),
                                ("second", X_phase2),
                                ("third", X_phase3)]:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []

        for train_idx, test_idx in kfold.split(X_phase):
            X_train, X_test = X_phase[train_idx], X_phase[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SymbolicRegressor(
                population_size=1000,  # 使用较小的种群大小以加快交叉验证
                generations=20,
                function_set=['add', 'sub', 'mul'],
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                parsimony_coefficient=0.05,
                random_state=42,
                verbose=0  # 关闭详细输出
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)

        avg_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        print(f"{phase_name}: average_RMSE = {avg_rmse:.4f} (±{std_rmse:.4f})")
