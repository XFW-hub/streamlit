# -*- coding: utf-8 -*-
"""
蓝莓产量分析全流程系统 v2（项目0227）
- 修复中文乱码（多编码读取 + 页面/图表字体）
- 无新上传时默认展示 data.csv
"""
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest
import warnings
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# ---------------------- 中文乱码修复：多编码读取 CSV ----------------------
def read_csv_safe(path_or_file, is_path=True):
    """按多种编码尝试读取 CSV，避免中文乱码。支持文件路径或上传的 file-like 对象。"""
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin1']
    if is_path:
        for enc in encodings:
            try:
                return pd.read_csv(path_or_file, encoding=enc)
            except (UnicodeDecodeError, Exception):
                continue
        return None
    else:
        raw = path_or_file.read()
        if isinstance(raw, bytes):
            for enc in encodings:
                try:
                    text = raw.decode(enc)
                    return pd.read_csv(io.StringIO(text))
                except (UnicodeDecodeError, Exception):
                    continue
        else:
            for enc in encodings:
                try:
                    path_or_file.seek(0)
                    return pd.read_csv(path_or_file, encoding=enc)
                except (UnicodeDecodeError, Exception):
                    continue
        return None

def get_default_data():
    """无新上传时优先使用内置的 data.csv；若无内置则从脚本同目录/项目路径/当前目录读 data.csv。返回 (df, source_label)。"""
    try:
        from _data_embedded import get_embedded_data
        df = get_embedded_data()
        if df is not None and not df.empty:
            return df, "内置 data.csv（无需外部文件）"
    except Exception:
        pass
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = r'D:\桌面\毕业论文\项目0227'
    candidates = [
        os.path.join(script_dir, 'data.csv'),
        os.path.join(default_path, 'data.csv'),
        os.path.join(os.getcwd(), 'data.csv'),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            df = read_csv_safe(path, is_path=True)
            if df is not None and not df.empty:
                return df, "默认 data.csv（来自文件）"
    return None, None

# ---------------------- 1. 页面配置与绘图风格 ----------------------
st.set_page_config(
    page_title="蓝莓产量分析全流程系统",
    page_icon="🫐",
    layout="wide"
)

# 中文乱码修复：整页使用支持中文的字体
st.markdown("""
<style>
    body, .stApp, [data-testid="stAppViewContainer"], p, span, div, label, input {
        font-family: "Microsoft YaHei", "微软雅黑", "SimHei", "SimSun", sans-serif !important;
    }
    .stMetric label { font-family: "Microsoft YaHei", "SimHei", sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# 图表中文：优先微软雅黑/黑体，减少 matplotlib 中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.alpha'] = 0.4
plt.rcParams['axes.linewidth'] = 1.0
COLOR_PALETTE = ['#2E86AB', '#E94B3C', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']
sns.set_palette(COLOR_PALETTE)

DEFAULT_DATA_PATH = r'D:\桌面\毕业论文\项目0227'
FEATURE_ORDER = [
    'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
    'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds', 'yield'
]
NORMAL_APPROX = [
    'clonesize', 'andrena', 'osmia',
    'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays', 'AverageRainingDays',
    'fruitset', 'fruitmass', 'seeds', 'yield'
]
NON_NORMAL = ['honeybee', 'bumbles']

# ---------------------- 2. 会话状态初始化 ----------------------
def init_session():
    for key, val in [
        ('df', None), ('train_df', None), ('test_df', None), ('train_clean', None),
        ('numeric_cols', []), ('scaler', None), ('pca', None), ('selector', None),
        ('X_train_pca', None), ('X_test_pca', None), ('y_train', None), ('y_test', None),
        ('best_model', None), ('best_model_name', None), ('model_trained', False),
        ('fill_rules', {}), ('train_stats', {}), ('outlier_done', False), ('dim_done', False),
        ('pred_feature_cols', []), ('y_pred', None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ---------------------- 3. 数据加载：无新上传时优先使用内置 data.csv ----------------------
default_df, default_label = get_default_data()
uploaded_file = st.file_uploader("选择 CSV 蓝莓数据集（不选则使用**内置 data.csv**）", type="csv")

if uploaded_file is not None:
    df = read_csv_safe(uploaded_file, is_path=False)
    data_source = "上传文件"
else:
    df = default_df
    data_source = default_label if default_label else "无可用数据"

st.title("🫐 蓝莓产量分析全流程系统（项目0227 v2）")
st.markdown("上传数据或使用默认数据 → 划分 → 描述统计/箱线图 → 正态性/异常值 → 相关性 → 降维 → 多模型对比 → 预测")

if df is None or df.empty:
    st.warning("未上传文件且未找到可用的默认 data.csv，请上传 CSV 或将 data.csv 放在脚本同目录/项目目录下。")
    st.stop()

st.success(f"当前使用：**{data_source}**（共 {len(df)} 行）")

df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
df = df.drop(["id"], axis=1, errors='ignore')
st.session_state.df = df

numeric_cols = [c for c in FEATURE_ORDER if c in df.columns]
if 'yield' in numeric_cols and len(numeric_cols) > 1:
    feature_cols = [c for c in numeric_cols if c != 'yield']
else:
    feature_cols = numeric_cols
st.session_state.numeric_cols = numeric_cols

st.subheader("1️⃣ 数据预览与划分")
col_a, col_b = st.columns(2)
with col_a:
    st.dataframe(df.head(10), use_container_width=True)
with col_b:
    test_ratio = st.slider("测试集比例", 0.1, 0.4, 0.3, 0.05)
    if st.button("划分训练集/测试集", type="primary"):
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=42, shuffle=True
        )
        st.session_state.train_df = train_df
        st.session_state.test_df = test_df
        st.success(f"划分完成：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")

train_df = st.session_state.train_df
test_df = st.session_state.test_df
if train_df is None or test_df is None:
    st.stop()

# ---------------------- 4. 描述性统计 ----------------------
st.subheader("2️⃣ 描述性统计")
desc = df[feature_cols].describe().T
desc['中位数'] = [round(df[col].median(), 4) for col in feature_cols]
desc['偏度'] = [round(df[col].skew(), 3) for col in feature_cols]
desc['峰度'] = [round(df[col].kurt(), 3) for col in feature_cols]
desc['变异系数'] = [round(desc.loc[col, 'std'] / desc.loc[col, 'mean'], 4) if desc.loc[col, 'mean'] != 0 else 0 for col in feature_cols]
st.dataframe(desc.style.background_gradient(subset=['mean', 'std'], cmap='Blues').format("{:.4f}", subset=desc.columns), use_container_width=True)

# ---------------------- 5. 箱线图 ----------------------
st.subheader("3️⃣ 箱线图")
box_var = st.selectbox("选择变量绘制箱线图", options=feature_cols, index=0)
fig_box, ax_box = plt.subplots(figsize=(8, 4))
ax_box.boxplot(
    df[box_var].dropna(), patch_artist=True,
    boxprops=dict(facecolor='#2E86AB', color='#1a5276', linewidth=1.2),
    whiskerprops=dict(color='#1a5276'),
    capprops=dict(color='#1a5276'),
    medianprops=dict(color='#E94B3C', linewidth=2),
    flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=4, alpha=0.7)
)
ax_box.set_ylabel("数值", fontsize=11)
ax_box.set_title(f"{box_var} 箱线图（中位数：{df[box_var].median():.2f}）", fontsize=12, fontweight='bold')
ax_box.grid(axis='y', alpha=0.4, linestyle='--')
ax_box.set_facecolor('#fafafa')
plt.tight_layout()
st.pyplot(fig_box)
plt.close()

# ---------------------- 6. 直方图 + 核密度 + Q-Q ----------------------
st.subheader("4️⃣ 正态性可视化（直方图 / 核密度 / Q-Q）")
norm_var = st.selectbox("选择变量", options=feature_cols, index=0, key="norm_var")
col1, col2 = st.columns(2)
with col1:
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    data = train_df[norm_var].dropna()
    ax_hist.hist(data, bins=28, density=True, alpha=0.7, color='#4CAF50', edgecolor='white', linewidth=0.5)
    data.plot(kind='kde', ax=ax_hist, color='#E94B3C', linewidth=2, label='核密度')
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 200)
    ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'b--', linewidth=1.5, label='正态拟合')
    ax_hist.set_title(f'{norm_var} 分布', fontsize=12, fontweight='bold')
    ax_hist.set_xlabel(norm_var, fontsize=10)
    ax_hist.set_ylabel('密度', fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(alpha=0.4, linestyle='--')
    ax_hist.set_facecolor('#fafafa')
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close()
with col2:
    fig_qq, ax_qq = plt.subplots(figsize=(7, 4))
    stats.probplot(train_df[norm_var].dropna(), plot=ax_qq, rvalue=True)
    ax_qq.set_title(f'{norm_var} Q-Q 图', fontsize=12, fontweight='bold')
    ax_qq.set_xlabel('理论分位数', fontsize=10)
    ax_qq.set_ylabel('样本分位数', fontsize=10)
    ax_qq.grid(alpha=0.4, linestyle='--')
    ax_qq.set_facecolor('#fafafa')
    plt.tight_layout()
    st.pyplot(fig_qq)
    plt.close()

# ---------------------- 7. KS 与综合正态性 ----------------------
st.subheader("5️⃣ 正态性检验（K-S + 综合判断）")
norm_results = []
for col in feature_cols + (['yield'] if 'yield' in df.columns else []):
    if col not in train_df.columns:
        continue
    data = train_df[col].dropna()
    if len(data) < 3:
        continue
    skewness = data.skew()
    kurtosis = data.kurt()
    mu, std = data.mean(), data.std()
    ks_stat, p_value = kstest(data, 'norm', args=(mu, std))
    (_, _), (_, _, r_sq) = stats.probplot(data, dist='norm', plot=None)
    qq_fit = "非常好" if r_sq > 0.95 else ("好" if r_sq > 0.9 else ("中等" if r_sq > 0.8 else "差"))
    if abs(skewness) < 1 and abs(kurtosis) < 1 and r_sq > 0.9:
        judge = "正态"
    elif abs(skewness) < 2 and abs(kurtosis) < 2 and r_sq > 0.8:
        judge = "近似正态"
    else:
        judge = "非正态"
    norm_results.append({
        '变量': col, '偏度': round(skewness, 3), '峰度': round(kurtosis, 3),
        'KS统计量': round(ks_stat, 4), 'K-S p值': round(p_value, 4),
        'QQ图R²': round(r_sq, 4), 'QQ拟合': qq_fit, '综合判断': judge
    })
st.dataframe(pd.DataFrame(norm_results), use_container_width=True)

# ---------------------- 8. 异常值处理 ----------------------
st.subheader("6️⃣ 异常值处理（3σ / IQR）")
if st.button("执行异常值处理（训练集删除，测试集仅标记）"):
    train_outlier_mask = pd.DataFrame(False, index=train_df.index, columns=train_df.columns)
    train_stats = {}
    for col in NORMAL_APPROX:
        if col not in train_df.columns:
            continue
        data = train_df[col].dropna()
        if len(data) == 0:
            continue
        mean, std = data.mean(), data.std()
        train_stats[col] = {"type": "normal", "mean": mean, "std": std}
        lo, hi = mean - 3 * std, mean + 3 * std
        train_outlier_mask.loc[data.index, col] = (data < lo) | (data > hi)
    for col in NON_NORMAL:
        if col not in train_df.columns:
            continue
        data = train_df[col].dropna()
        if len(data) == 0:
            continue
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        train_stats[col] = {"type": "non_normal", "Q1": Q1, "Q3": Q3, "IQR": IQR}
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        train_outlier_mask.loc[data.index, col] = (data < lo) | (data > hi)
    drop_mask = train_outlier_mask.any(axis=1)
    train_clean = train_df[~drop_mask].copy()
    st.session_state.train_clean = train_clean
    st.session_state.train_stats = train_stats
    st.session_state.outlier_done = True
    st.success(f"训练集清洗完成：删除 {drop_mask.sum()} 行，保留 {len(train_clean)} 行")

train_clean = st.session_state.train_clean
if train_clean is None:
    train_clean = train_df
corr_df = train_clean

# ---------------------- 9. 相关性热力图 ----------------------
st.subheader("7️⃣ 特征相关性热力图")
corr_cols = [c for c in feature_cols + ['yield'] if c in corr_df.columns]
corr_matrix = corr_df[corr_cols].dropna().corr().round(2)
fig_corr, ax_corr = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(
    corr_matrix, ax=ax_corr, cmap=cmap, annot=True, fmt='.2f',
    vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Pearson 相关系数'},
    annot_kws={'size': 8}
)
ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=9)
ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=9)
ax_corr.set_title("特征相关性热力图（训练集）", fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
st.pyplot(fig_corr)
plt.close()

# ---------------------- 10. 降维 ----------------------
st.subheader("8️⃣ 降维（标准化 → PCA / SelectKBest）")
target_col = 'yield'
X_tr = corr_df.drop(target_col, axis=1, errors='ignore')
X_tr = X_tr[[c for c in FEATURE_ORDER if c in X_tr.columns and c != target_col]]
y_tr = corr_df[target_col]
X_te = test_df.drop(target_col, axis=1, errors='ignore')
X_te = X_te[X_tr.columns]
y_te = test_df[target_col]
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
st.session_state.scaler = scaler
st.session_state.fill_rules = {c: float(X_tr[c].median()) for c in X_tr.columns}

dim_method = st.radio("降维方式", ["PCA（保留 80% 方差）", "SelectKBest（Top 10）"], index=0)
if st.button("执行降维"):
    if "PCA" in dim_method:
        pca = PCA(n_components=0.8, random_state=42)
        X_train_pca = pca.fit_transform(X_tr_s)
        X_test_pca = pca.transform(X_te_s)
        st.session_state.pca = pca
        st.session_state.selector = None
    else:
        k_best = min(10, X_tr_s.shape[1])
        selector = SelectKBest(score_func=f_regression, k=k_best)
        X_train_pca = selector.fit_transform(X_tr_s, y_tr)
        X_test_pca = selector.transform(X_te_s)
        st.session_state.selector = selector
        st.session_state.pca = None
    st.session_state.X_train_pca = X_train_pca
    st.session_state.X_test_pca = X_test_pca
    st.session_state.y_train = y_tr.values
    st.session_state.y_test = y_te.values
    st.session_state.pred_feature_cols = list(X_tr.columns)
    st.session_state.dim_done = True
    st.success(f"降维完成：训练 {X_train_pca.shape}，测试 {X_test_pca.shape}")

if st.session_state.dim_done:
    X_train_pca = st.session_state.X_train_pca
    X_test_pca = st.session_state.X_test_pca
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    pca_obj = st.session_state.pca
    if pca_obj is not None:
        fig_pca, ax_pca = plt.subplots(figsize=(8, 4))
        ax_pca.plot(range(1, len(pca_obj.explained_variance_ratio_) + 1),
                    np.cumsum(pca_obj.explained_variance_ratio_), 'o-', color='#2E86AB', linewidth=2, markersize=8)
        ax_pca.set_xlabel("主成分数", fontsize=11)
        ax_pca.set_ylabel("累计解释方差比", fontsize=11)
        ax_pca.set_title("PCA 累计解释方差比", fontsize=12, fontweight='bold')
        ax_pca.grid(alpha=0.4)
        ax_pca.set_facecolor('#fafafa')
        plt.tight_layout()
        st.pyplot(fig_pca)
        plt.close()

# ---------------------- 11. 多模型对比与选优 ----------------------
st.subheader("9️⃣ 多模型 5 折 CV 与测试集验证")
if not st.session_state.dim_done:
    st.warning("请先完成「执行降维」再建模。")
else:
    X_train_pca = st.session_state.X_train_pca
    X_test_pca = st.session_state.X_test_pca
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    class ManualStackingRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, base_models, meta_model, cv=5):
            self.base_models = base_models
            self.meta_model = meta_model
            self.cv = cv
            self.fitted_base_models = {}
            self.cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        def fit(self, X, y):
            X = np.asarray(X)
            y = np.asarray(y).ravel()
            X_meta = np.zeros((X.shape[0], len(self.base_models)))
            for idx, (name, model) in enumerate(self.base_models.items()):
                fold_preds = np.zeros(X.shape[0])
                for tr_idx, val_idx in self.cv_splitter.split(X):
                    model.fit(X[tr_idx], y[tr_idx])
                    fold_preds[val_idx] = model.predict(X[val_idx])
                X_meta[:, idx] = fold_preds
                self.fitted_base_models[name] = model.fit(X, y)
            self.meta_model.fit(X_meta, y)
            self.is_fitted_ = True
            return self
        def predict(self, X):
            X = np.asarray(X)
            X_meta = np.column_stack([m.predict(X) for m in self.fitted_base_models.values()])
            return self.meta_model.predict(X_meta)

    base_models = {
        'xgb': XGBRegressor(n_estimators=100, random_state=42) if HAS_XGB else RandomForestRegressor(n_estimators=100, random_state=42),
        'lgb': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1) if HAS_LGB else RandomForestRegressor(n_estimators=100, random_state=43),
        'rf': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    meta_model = LassoCV(cv=5, max_iter=10000)
    stacking = ManualStackingRegressor(base_models=base_models, meta_model=meta_model, cv=5) if (HAS_XGB or HAS_LGB) else None

    models_dict = {
        "LassoCV": LassoCV(cv=5, max_iter=10000),
        "随机森林": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "线性回归": LinearRegression(),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42),
    }
    if HAS_XGB:
        models_dict["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
    if HAS_LGB:
        models_dict["LightGBM"] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31, subsample=0.8, random_state=42, verbose=-1)
    if stacking is not None:
        models_dict["Stacking"] = stacking

    if st.button("运行 5 折交叉验证并选优"):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = {}
        with st.spinner("正在运行各模型 CV..."):
            for name, model in models_dict.items():
                t0 = time.time()
                r2_scores = cross_val_score(model, X_train_pca, y_train, cv=kf, scoring='r2')
                mae_scores = -cross_val_score(model, X_train_pca, y_train, cv=kf, scoring='neg_mean_absolute_error')
                cv_res[name] = {
                    "CV平均R²": round(np.mean(r2_scores), 4),
                    "CV R²标准差": round(np.std(r2_scores), 4),
                    "CV平均MAE": round(np.mean(mae_scores), 4),
                    "CV耗时(s)": round(time.time() - t0, 2)
                }
        df_cv = pd.DataFrame(cv_res).T.reset_index().rename(columns={'index': '模型'})
        df_cv = df_cv.sort_values(by=["CV平均R²", "CV R²标准差"], ascending=[False, True])
        best_name = df_cv.iloc[0]["模型"]
        best_model = models_dict[best_name]
        best_model.fit(X_train_pca, y_train)
        y_pred = best_model.predict(X_test_pca)
        test_r2 = round(r2_score(y_test, y_pred), 4)
        test_mae = round(mean_absolute_error(y_test, y_pred), 4)
        test_rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        st.session_state.best_model = best_model
        st.session_state.best_model_name = best_name
        st.session_state.model_trained = True
        st.session_state.y_pred = y_pred

        st.metric("最优模型", best_name)
        st.dataframe(df_cv, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("测试集 R²", test_r2)
        c2.metric("测试集 MAE", test_mae)
        c3.metric("测试集 RMSE", test_rmse)

        fig_cv, ax_cv = plt.subplots(figsize=(10, 5))
        x_pos = np.arange(len(df_cv))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_cv)))
        ax_cv.bar(x_pos, df_cv["CV平均R²"].values, yerr=df_cv["CV R²标准差"].values,
                 capsize=5, color=colors, edgecolor='#333', linewidth=0.8)
        ax_cv.set_xticks(x_pos)
        ax_cv.set_xticklabels(df_cv["模型"].values, rotation=35, ha='right')
        ax_cv.set_ylabel("CV 平均 R²", fontsize=11)
        ax_cv.set_title("各模型 5 折交叉验证 R² 对比", fontsize=13, fontweight='bold')
        ax_cv.set_ylim(0, 1.05)
        ax_cv.grid(axis='y', alpha=0.4)
        ax_cv.set_facecolor('#fafafa')
        plt.tight_layout()
        st.pyplot(fig_cv)
        plt.close()

        fig_scatter, ax_scatter = plt.subplots(figsize=(7, 5))
        ax_scatter.scatter(y_test, y_pred, alpha=0.7, c='#2E86AB', edgecolors='white', s=60)
        mi, ma = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax_scatter.plot([mi, ma], [mi, ma], 'r--', lw=2, label='理想线')
        ax_scatter.set_xlabel("真实产量", fontsize=11)
        ax_scatter.set_ylabel("预测产量", fontsize=11)
        ax_scatter.set_title(f"{best_name} 测试集：真实 vs 预测", fontsize=12, fontweight='bold')
        ax_scatter.legend()
        ax_scatter.grid(alpha=0.4)
        ax_scatter.set_facecolor('#fafafa')
        plt.tight_layout()
        st.pyplot(fig_scatter)
        plt.close()

# ---------------------- 12. 单样本预测 ----------------------
st.subheader("🔟 单样本产量预测")
if not st.session_state.model_trained:
    st.warning("请先完成「运行 5 折交叉验证并选优」再进行预测。")
else:
    feat_list = getattr(st.session_state, 'pred_feature_cols', None) or [c for c in FEATURE_ORDER if c in (st.session_state.train_clean or st.session_state.train_df).columns and c != 'yield']
    st.markdown(f"输入 {len(feat_list)} 个特征值（将自动标准化并降维后预测）。")
    input_vals = {}
    cols_per_row = 3
    _train = st.session_state.train_clean if st.session_state.train_clean is not None else st.session_state.train_df
    X_tr_ref = _train[feat_list] if (_train is not None and feat_list) else pd.DataFrame()
    for i in range(0, len(feat_list), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col in enumerate(feat_list[i:i + cols_per_row]):
            default = float(X_tr_ref[col].median()) if (len(X_tr_ref) > 0 and col in X_tr_ref.columns) else 0.0
            input_vals[col] = row_cols[j].number_input(col, value=default, step=0.01, key=f"pred_{col}")

    if st.button("预测产量", type="primary"):
        try:
            X_in = pd.DataFrame([input_vals], columns=feat_list)
            X_s = st.session_state.scaler.transform(X_in)
            pca_obj = st.session_state.pca
            sel = st.session_state.selector
            if pca_obj is not None:
                X_s = pca_obj.transform(X_s)
            elif sel is not None:
                X_s = sel.transform(X_s)
            pred = st.session_state.best_model.predict(X_s)[0]
            st.success(f"预测产量：**{round(pred, 2)}**")
        except Exception as e:
            st.error(f"预测失败：{e}")
