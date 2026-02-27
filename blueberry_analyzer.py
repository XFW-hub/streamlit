import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------- 1. 页面基础配置 ----------------------
st.set_page_config(
    page_title="蓝莓产量分析与预测系统",
    page_icon="🫐",
    layout="wide"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 标题与说明 ----------------------
st.title("🫐 蓝莓产量分析与预测系统")
st.markdown("### 上传数据集 → 选择变量 → 查看直方图/箱线图 → 建模预测")

# ---------------------- 3. 初始化会话状态（替代global，更安全） ----------------------
# 用Streamlit的session_state存储关键变量，避免全局变量问题
if 'numeric_cols' not in st.session_state:
    st.session_state.numeric_cols = []
if 'fill_rules' not in st.session_state:
    st.session_state.fill_rules = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'model' not in st.session_state:
    st.session_state.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ---------------------- 4. 数据上传 + 去Unnamed列 ----------------------
uploaded_file = st.file_uploader("选择CSV格式的蓝莓数据集", type="csv")
if uploaded_file is not None:
    # 读取数据并剔除Unnamed列
    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    st.success("✅ 数据上传成功（已自动剔除无意义的Unnamed列）！")

    # 显示数据预览
    st.subheader("1. 数据预览")
    st.dataframe(df.head(10), use_container_width=True)

    # ---------------------- 核心修改：用session_state存储特征列表（无global） ----------------------
    # 筛选数值型列（训练/预测共用）
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # 移除yield列（标签列，非特征）
    if 'yield' in numeric_cols:
        numeric_cols.remove('yield')
    # 存入session_state，后续所有模块共用
    st.session_state.numeric_cols = numeric_cols

    if not numeric_cols:
        st.error("⚠️ 数据中无数值型特征变量，无法进行统计和建模！")
    else:
        st.info(f"🔍 检测到 {len(numeric_cols)} 个数值型特征：{', '.join(numeric_cols)}")

        # ---------------------- 5. 描述性统计 ----------------------
        st.subheader("2. 描述性统计结果")
        tab1, tab2 = st.tabs(["基础统计", "缺失值/异常值"])

        with tab1:
            # 特征列的描述性统计（不含yield）
            desc_stats = df[numeric_cols].describe().T
            desc_stats['中位数'] = [round(df[col].median(), 4) for col in numeric_cols]
            desc_stats['变异系数'] = [round(desc_stats.loc[col, 'std'] / desc_stats.loc[col, 'mean'], 4)
                                  if desc_stats.loc[col, 'mean'] != 0 else 0
                                  for col in numeric_cols]
            st.dataframe(desc_stats, use_container_width=True)

        with tab2:
            # 缺失值统计
            missing_stats = pd.DataFrame({
                "缺失值数量": df.isnull().sum(),
                "缺失率(%)": [round((v / len(df) * 100), 4) for v in df.isnull().sum()]
            })
            st.subheader("缺失值统计")
            st.dataframe(missing_stats, use_container_width=True)

            # 异常值统计（IQR法）
            st.subheader("异常值统计（IQR法）")
            outlier_stats = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_num = len(df[(df[col] < lower) | (df[col] > upper)])
                outlier_stats[col] = {
                    "异常值数量": outlier_num,
                    "异常值占比(%)": round(outlier_num / len(df) * 100, 4),
                    "下界": round(lower, 4),
                    "上界": round(upper, 4)
                }
            outlier_df = pd.DataFrame(outlier_stats).T
            st.dataframe(outlier_df, use_container_width=True)

        # ---------------------- 6. 变量可视化（直方图+箱线图选择） ----------------------
        st.subheader("3. 变量可视化（直方图+箱线图）")
        selected_var = st.selectbox("请选择要查看的变量", options=numeric_cols, index=0)

        if st.button("生成直方图 & 箱线图", type="primary"):
            # 双列布局展示图表
            col1, col2 = st.columns(2)
            # 直方图
            with col1:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.histplot(df[selected_var].dropna(), kde=True, bins=30, color='skyblue', ax=ax1)
                ax1.set_title(f"{selected_var} 分布直方图", fontsize=12, fontweight='bold')
                ax1.set_xlabel(selected_var, fontsize=10)
                ax1.set_ylabel("频数", fontsize=10)
                ax1.grid(axis='y', alpha=0.3)
                st.pyplot(fig1)
            # 箱线图
            with col2:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=df[selected_var].dropna(), color='lightcoral', ax=ax2,
                            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 5})
                median = df[selected_var].median()
                ax2.set_title(f"{selected_var} 箱线图（中位数：{median:.2f}）", fontsize=12, fontweight='bold')
                ax2.set_xlabel(selected_var, fontsize=10)
                ax2.grid(axis='x', alpha=0.3)
                st.pyplot(fig2)

        # ---------------------- 7. 相关性热力图 ----------------------
        st.subheader("4. 特征相关性热力图")
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title("特征相关性热力图", fontsize=14, fontweight='bold')
        st.pyplot(fig)

        # ---------------------- 8. 自动建模与预测（无global，用session_state） ----------------------
        if 'yield' in df.columns and st.checkbox("🔧 开启自动建模与产量预测", value=False):
            st.subheader("5. 自动建模与产量预测")

            # 数据预处理（无泄露流程）
            with st.spinner("正在进行数据预处理..."):
                df_labeled = df[df['yield'].notna()].copy()
                X = df_labeled[numeric_cols]  # 用所有特征
                y = df_labeled['yield']

                # 拆分训练集/验证集
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                # 缺失值填充规则（存入session_state）
                fill_rules = {col: round(X_train[col].median(), 4) for col in numeric_cols}
                st.session_state.fill_rules = fill_rules
                X_train_filled = X_train.fillna(fill_rules)
                X_val_filled = X_val.fillna(fill_rules)

                # 标准化（存入session_state）
                scaler = StandardScaler()
                scaler.fit(X_train_filled)  # 仅训练集拟合
                st.session_state.scaler = scaler
                X_train_scaled = scaler.transform(X_train_filled)
                X_val_scaled = scaler.transform(X_val_filled)

            # 选择模型
            model_type = st.radio("选择建模算法", ["随机森林回归", "线性回归"], index=0)
            if st.button("开始训练模型", type="secondary"):
                with st.spinner("正在训练模型..."):
                    # 训练模型（存入session_state）
                    if model_type == "随机森林回归":
                        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                    else:
                        model = LinearRegression()
                    model.fit(X_train_scaled, y_train)
                    st.session_state.model = model
                    st.session_state.model_trained = True  # 标记模型已训练

                    # 模型评估
                    y_val_pred = model.predict(X_val_scaled)
                    r2 = round(r2_score(y_val, y_val_pred), 4)
                    mae = round(mean_absolute_error(y_val, y_val_pred), 4)
                    rmse = round(np.sqrt(mean_squared_error(y_val, y_val_pred)), 4)

                    # 展示评估结果
                    st.subheader("模型评估结果")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² 得分", f"{r2}")
                    col2.metric("MAE（平均绝对误差）", f"{mae}")
                    col3.metric("RMSE（均方根误差）", f"{rmse}")

                    # 预测vs真实值可视化
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_val, y_val_pred, alpha=0.6, color='orange')
                    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
                    ax.set_xlabel("真实产量", fontsize=10)
                    ax.set_ylabel("预测产量", fontsize=10)
                    ax.set_title(f"{model_type} 预测效果", fontsize=12, fontweight='bold')
                    st.pyplot(fig)

                    # 特征重要性（仅随机森林）
                    if model_type == "随机森林回归":
                        st.subheader("特征重要性排序")
                        importance = pd.DataFrame({
                            "特征": numeric_cols,
                            "重要性": [round(v, 4) for v in model.feature_importances_]
                        }).sort_values("重要性", ascending=False)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x="重要性", y="特征", data=importance.head(10), ax=ax)
                        ax.set_title("Top10 特征重要性", fontsize=12, fontweight='bold')
                        st.pyplot(fig)

            # ---------------------- 9. 单样本预测（所有特征输入） ----------------------
            st.subheader("6. 单样本产量预测")
            if not st.session_state.model_trained:
                st.warning("⚠️ 请先训练模型后再进行预测！")
            else:
                st.markdown(f"⚠️ 请输入所有 {len(numeric_cols)} 个特征值（确保和训练时一致）")

                # 生成所有特征的输入框（分栏展示）
                input_features = {}
                cols_per_row = 3
                rows = [numeric_cols[i:i + cols_per_row] for i in range(0, len(numeric_cols), cols_per_row)]

                for row in rows:
                    input_cols = st.columns(cols_per_row)
                    for idx, col in enumerate(row):
                        default_val = round(df[col].median(), 4)
                        input_features[col] = input_cols[idx].number_input(
                            f"输入{col}值",
                            value=float(default_val),
                            step=0.01,
                            key=col
                        )

                # 转换为DataFrame（特征顺序一致）
                input_df = pd.DataFrame([input_features], columns=numeric_cols)
                # 填充缺失值（用session_state中的规则）
                input_df = input_df.fillna(st.session_state.fill_rules)
                # 标准化（用session_state中的scaler）
                input_scaled = st.session_state.scaler.transform(input_df)

                # 预测按钮
                if st.button("预测产量", type="primary"):
                    try:
                        pred_yield = st.session_state.model.predict(input_scaled)[0]
                        st.success(f"✅ 预测产量为：{round(pred_yield, 2)}")
                    except Exception as e:
                        st.error(f"❌ 预测失败：{str(e)}")

else:
    st.info("💡 请先上传CSV格式的蓝莓数据集（支持特征：clonesize、honeybee、bumbles、osmia、fruitset、yield等）")