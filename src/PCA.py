import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from astropy.cosmology import FlatLambdaCDM


def load_and_preprocess_data(file_path, r1=0.5, r2=1.2):
    """
    加载数据并进行预处理（与RF代码相同的处理流程）
    """
    df = pd.read_csv(file_path)

    # 选择红移范围
    selected = df[((df['redshift'] > r1) & (df['redshift'] < r2))]

    # 计算绝对星等（与RF代码相同）
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    d_l = (cosmo.luminosity_distance(selected['redshift'])).value

    M_u = selected['u_app'] - 5 * np.log10(d_l) - 25
    M_g = selected['g_app'] - 5 * np.log10(d_l) - 25
    M_r = selected['r_app'] - 5 * np.log10(d_l) - 25
    M_i = selected['i_app'] - 5 * np.log10(d_l) - 25
    M_z = selected['z_app'] - 5 * np.log10(d_l) - 25

    # 计算红移相关特征
    redshift = selected['redshift'].values
    redshift2 = selected['redshift'] ** 2
    redshift3 = selected['redshift'] + 1
    redshift4 = 1 / (selected['redshift'] + 1)

    # 计算颜色
    color_ug = M_u - M_g
    color_gr = M_g - M_r
    color_ri = M_r - M_i
    color_iz = M_i - M_z

    # 创建包含所有特征的数据框
    features_df = pd.DataFrame({
        'M_i': M_i.values,
        'M_z': M_z.values,
        'M_r': M_r.values,
        'M_g': M_g.values,
        'M_u': M_u.values,
        'color_ug': color_ug.values,
        'color_gr': color_gr.values,
        'color_ri': color_ri.values,
        'color_iz': color_iz.values,
        'redshift': redshift,
        'redshift2': redshift2.values,
        'redshift3': redshift3.values,
        'redshift4': redshift4.values
    })

    return features_df


def run_pca_analysis(data, feature_columns, n_components=None):
    """
    Complete PCA analysis workflow
    """
    # 1. Data preparation
    X = data[feature_columns].copy()

    print("=== PCA Analysis Started ===")
    print(f"Original data shape: {X.shape}")
    print(f"Number of features: {len(feature_columns)}")
    print("\nFeature list:")
    for i, col in enumerate(feature_columns, 1):
        print(f"{i}. {col}")

    # 2. Data standardization (essential before PCA!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Perform PCA
    if n_components is None:
        pca = PCA()  # Automatically compute all components
    else:
        pca = PCA(n_components=n_components)

    principal_components = pca.fit_transform(X_scaled)

    return pca, principal_components, X_scaled


def plot_pca_results(pca, feature_columns, principal_components):
    """
    Visualize PCA results
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Explained variance ratio (Scree plot)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    axes[0, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
                   alpha=0.6, color='skyblue', label='Individual component variance')
    axes[0, 0].plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
                    marker='o', color='red', label='Cumulative variance')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Explained Variance Ratio')
    axes[0, 0].set_title('PCA Explained Variance (Scree Plot)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Mark component that reaches 95% variance explained
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    axes[0, 0].axvline(x=n_components_95, color='green', linestyle='--',
                       label=f'95% variance: {n_components_95} components')
    axes[0, 0].legend()

    # 2. Scatter plot of first two principal components
    axes[0, 1].scatter(principal_components[:, 0], principal_components[:, 1],
                       alpha=0.6, color='blue')
    axes[0, 1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    axes[0, 1].set_title('Distribution of First Two Principal Components')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Principal component loadings (feature weights) heatmap
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Show loadings for first 10 components only
    n_show = min(10, len(feature_columns))
    loadings_df = pd.DataFrame(loadings[:, :n_show],
                               index=feature_columns,
                               columns=[f'PC{i + 1}' for i in range(n_show)])

    sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0,
                ax=axes[1, 0], fmt='.2f')
    axes[1, 0].set_title('Principal Component Loadings Matrix')

    # 4. Feature contributions in PC space (Variable plot)
    for i, feature in enumerate(feature_columns):
        axes[1, 1].arrow(0, 0, loadings[i, 0], loadings[i, 1],
                         head_width=0.05, head_length=0.05, fc='k', ec='k')
        axes[1, 1].text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15,
                        feature, color='darkred', ha='center', va='center')

    # Add explained variance circle
    circle = plt.Circle((0, 0), 1, fill=False, color='blue', linestyle='--')
    axes[1, 1].add_artist(circle)
    axes[1, 1].set_xlim(-1.5, 1.5)
    axes[1, 1].set_ylim(-1.5, 1.5)
    axes[1, 1].set_xlabel('PC1')
    axes[1, 1].set_ylabel('PC2')
    axes[1, 1].set_title('Feature Directions in PC Space (Variable Plot)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return n_components_95, loadings_df


def print_pca_summary(pca, feature_columns, n_components_95, loadings_df):
    """
    Print PCA analysis summary
    """
    print("\n=== PCA Analysis Summary ===")
    print(f"Original number of features: {len(feature_columns)}")
    print(f"Number of components needed for 95% variance: {n_components_95}")
    print(f"Dimensionality reduction: {len(feature_columns)} → {n_components_95}")

    print(f"\nVariance explained by each component:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:10], 1):
        print(f"PC{i}: {ratio:.3f} ({ratio:.1%})")

    print(f"\nTop features for first 5 principal components:")
    for i in range(min(5, len(feature_columns))):
        pc_name = f'PC{i + 1}'
        # Find top 3 features with highest absolute loadings
        top_features = loadings_df[pc_name].abs().nlargest(3)
        print(
            f"{pc_name}: {', '.join([f'{feat}({loadings_df.loc[feat, pc_name]:.2f})' for feat in top_features.index])}")


# def pca_by_redshift_range(features_df, redshift_col='redshift', redshift_ranges=None):
#     """
#     按不同红移区间分别进行PCA分析
#     """
#     if redshift_ranges is None:
#         redshift_ranges = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.2)]
#
#     feature_columns = [col for col in features_df.columns if col != redshift_col]
#
#     for low, high in redshift_ranges:
#         print(f"\n{'=' * 60}")
#         print(f"PCA Analysis for Redshift Range {low}-{high}")
#         print(f"{'=' * 60}")
#
#         # 选择该红移区间的数据
#         mask = (features_df[redshift_col] >= low) & (features_df[redshift_col] < high)
#         subset = features_df[mask]
#
#         if len(subset) < 50:  # 如果数据点太少，跳过
#             print(f"Not enough data points ({len(subset)}), skipping...")
#             continue
#
#         print(f"Number of galaxies in this range: {len(subset)}")
#
#         # 运行PCA分析
#         pca, principal_components, X_scaled = run_pca_analysis(subset, feature_columns)
#         n_components_95, loadings_df = plot_pca_results(pca, feature_columns, principal_components)
#         print_pca_summary(pca, feature_columns, n_components_95, loadings_df)


def main():
    """
    主函数 - 运行完整的PCA分析
    """
    # 参数设置（与你的RF代码相同）
    file_path = r'GALFORM_Vmax_perturbed_i225.csv'
    r1, r2 = 0.5, 1.2

    print("Loading and preprocessing data...")
    features_df = load_and_preprocess_data(file_path, r1, r2)

    # 特征列表（与RF代码相同）
    feature_columns = ['M_i', 'M_z', 'M_r', 'M_g', 'M_u',
                       'color_ug', 'color_gr', 'color_ri', 'color_iz',
                       'redshift', 'redshift2', 'redshift3', 'redshift4']

    # 选项1: 在整个红移范围运行PCA
    print("Running PCA on entire redshift range...")
    pca, principal_components, X_scaled = run_pca_analysis(features_df, feature_columns)
    n_components_95, loadings_df = plot_pca_results(pca, feature_columns, principal_components)
    print_pca_summary(pca, feature_columns, n_components_95, loadings_df)

    # 选项2: 按红移区间分别运行PCA
    print("\n" + "=" * 80)
    print("Now running PCA analysis by redshift ranges...")
    print("=" * 80)
    # pca_by_redshift_range(features_df)

    # 保存降维后的数据
    n_optimal = n_components_95
    pca_reduced = PCA(n_components=n_optimal)
    X_reduced = pca_reduced.fit_transform(X_scaled)

    print(f"\n=== Dimensionality Reduction Results ===")
    print(f"Original feature dimension: {len(feature_columns)}")
    print(f"Reduced dimension: {n_optimal}")
    print(f"Compression ratio: {len(feature_columns) / n_optimal:.2f}x")
    print(f"New features (principal components) shape: {X_reduced.shape}")


if __name__ == "__main__":
    main()
