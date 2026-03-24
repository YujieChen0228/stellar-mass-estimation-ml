import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
# --- 放在 import 之后：添加一个兼容新旧版本 sklearn 的 RMSE 函数 ---
def rmse_metric(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    try:
        # 新版本支持 squared=False
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # 旧版本没有 squared 参数：先算 MSE，再开根号变 RMSE
        return mean_squared_error(y_true, y_pred) ** 0.5

warnings.filterwarnings('ignore')
np.random.seed(42)

# ---------- 可调参数 ----------
csv_path = 'GALFORM_Vmax_perturbed_i225.csv'
bin_width = 0.1            # 统一两种图的bin设置
redshift_bins = np.arange(0, 2.1, bin_width)
test_size = 0.2
n_estimators = 100
max_features = 'sqrt'
n_bootstraps = 50          # bootstrap次数（=0表示不做bootstrap）
# ----------------------------

cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
df = pd.read_csv(csv_path)

def build_features(sel):
    d_l = cosmo.luminosity_distance(sel['redshift']).value
    M_u = sel['u_app'] - 5*np.log10(d_l) - 25
    M_g = sel['g_app'] - 5*np.log10(d_l) - 25
    M_r = sel['r_app'] - 5*np.log10(d_l) - 25
    M_i = sel['i_app'] - 5*np.log10(d_l) - 25
    M_z = sel['z_app'] - 5*np.log10(d_l) - 25

    redshift = sel['redshift'].values
    redshift2 = sel['redshift']**2
    redshift3 = sel['redshift'] - 1
    redshift4 = 1/(sel['redshift']+1)

    log10m = np.log10(sel['mstar_tot'])

    color_ug = M_u - M_g
    color_gr = M_g - M_r
    color_ri = M_r - M_i
    color_iz = M_i - M_z

    X = np.column_stack((
        M_i.values, M_z.values, M_r.values, M_g.values, M_u.values,
        color_ug.values, color_gr.values, color_ri.values, color_iz.values,
        redshift, redshift2.values, redshift3.values, redshift4
    ))
    y = np.array(log10m)
    feat_names = ['M_i','M_z','M_r','M_g','M_u','color_ug','color_gr','color_ri',
                  'color_iz','redshift','redshift2','redshift3','redshift4']
    return X, y, feat_names

def single_run_ratio(X, y, feat_names, random_state=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,
                               n_jobs=-1, max_features=max_features)
    rf.fit(X_tr, y_tr)
    imp = rf.feature_importances_
    imp = imp/imp.sum()
    d = dict(zip(feat_names, imp))
    ratio = d['color_iz']/d['M_z']
    r2 = r2_score(y_te, rf.predict(X_te))
    rmse = rmse_metric(y_te, rf.predict(X_te))
    return ratio, r2, rmse

def bootstrap_ratio(X, y, feat_names, n_boot=50):
    ratios, r2s, rmses = [], [], []
    for b in range(n_boot):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=b)
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=b,
                                   n_jobs=-1, max_features=max_features)
        rf.fit(X_tr, y_tr)
        imp = rf.feature_importances_
        imp = imp/imp.sum()
        d = dict(zip(feat_names, imp))
        ratios.append(d['color_iz']/d['M_z'])
        y_pred = rf.predict(X_te)
        r2s.append(r2_score(y_te, y_pred))
        rmses.append(mean_squared_error(y_te, y_pred))
    ratios = np.array(ratios)
    return {
        'mean_ratio': ratios.mean(),
        'ci_low':     np.percentile(ratios, 2.5),
        'ci_high':    np.percentile(ratios, 97.5),
        'std_ratio':  ratios.std(),
        'mean_r2':    np.mean(r2s),
        'mean_rmse':  np.mean(rmses)
    }

rows = []
for i in range(len(redshift_bins)-1):
    z_low, z_high = redshift_bins[i], redshift_bins[i+1]
    sel = df[(df['redshift'] > z_low) & (df['redshift'] <= z_high)]
    if len(sel) < 1000:
        continue
    X, y, feat_names = build_features(sel)
    # 单次结果（原上面面板(a)）
    ratio_single, r2_single, rmse_single = single_run_ratio(X, y, feat_names, random_state=42)
    # bootstrap统计（原下面面板(b)）
    stats = bootstrap_ratio(X, y, feat_names, n_boot=n_bootstraps)
    rows.append({
        'z_center': (z_low+z_high)/2,
        'N': len(sel),
        'ratio_single': ratio_single,
        **stats
    })

res = pd.DataFrame(rows)
assert not res.empty, "No bins had enough samples."

# ----------- 合成单面板图：bootstrap±CI + 单次结果线 -----------
plt.figure(figsize=(8,5.6))
# CI带
plt.fill_between(res['z_center'], res['ci_low'], res['ci_high'], alpha=0.20, label='Bootstrapped 95% CI',)
# bootstrap均值
plt.plot(res['z_center'], res['mean_ratio'], 'o-', linewidth=2, markersize=6, label='Bootstrapped mean')
# 单次 run（原(a)）
plt.plot(res['z_center'], res['ratio_single'], 's--', linewidth=2, markersize=6, label='Single run')
# 参考线
plt.axhline(1.0, linestyle='--', alpha=0.7, label='Equal Importance')
plt.xlabel('Redshift (z)',fontsize=14)
plt.ylabel('Ratio: color_iz / M_z',fontsize=14)
plt.title(f'Feature-Importance Ratio vs. Redshift (bin={bin_width}, n_boot={n_bootstraps})',fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=14)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('feature_importance_ratio_combined.png', dpi=300)
plt.show()
