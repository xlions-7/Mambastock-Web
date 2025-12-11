import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

# 引入你的 Mamba 模型定义
# 确保 mamba.py 在同一目录下
try:
    from mamba import Mamba, MambaConfig
except ImportError:
    st.error("未找到 mamba.py，请确保该文件在同一目录下！")
    st.stop()

# ==========================================
# 1. 页面配置与 CSS 优化
# ==========================================
st.set_page_config(
    page_title="MambaStock 股价预测系统",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 2. 模型定义 (保持与之前一致)
# ==========================================
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, layer_num):
        super().__init__()
        self.config = MambaConfig(d_model=hidden_dim, n_layers=layer_num)
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.mamba = Mamba(self.config)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mamba(x)
        x = x[:, -1, :] 
        return self.decoder(x).flatten()

# ==========================================
# 3. 侧边栏：参数配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # Tushare Token
    ts_token = st.text_input("Tushare Token", value="e46f511c90393a9198ebd810f272cad660d392f3373aa6a546339c32", type="password")
    
    st.divider()
    
    # 股票设置
    stock_code = st.text_input("股票代码 (如 000001)", value="601988")
    start_date = st.date_input("开始日期", datetime.date(2018, 1, 1))
    
    st.divider()
    
    # 模型超参数
    epochs = st.number_input("训练轮数 (Epochs)", min_value=10, max_value=1000, value=100, step=10)
    lr = st.number_input("学习率 (LR)", value=0.005, format="%.4f")
    hidden_dim = st.selectbox("隐藏层维度", [16, 32, 64, 128], index=1)
    seq_len = st.slider("时间窗口 (Sequence Length)", 10, 60, 30, help="利用过去多少天的数据来预测下一天")
    
    run_btn = st.button("🚀 开始训练与预测", type="primary")

# ==========================================
# 4. 核心逻辑函数
# ==========================================

@st.cache_data(ttl=3600) # 缓存数据1小时，避免重复下载
def get_data(token, code, start_str):
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 自动补全后缀
    if not code.endswith(('.SH', '.SZ')):
        code += '.SH' if code.startswith('6') else '.SZ'
        
    df = ts.pro_bar(ts_code=code, adj='qfq', start_date=start_str)
    
    if df is None or df.empty:
        return None, code
        
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    
    # 添加指标
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df.dropna(inplace=True)
    
    return df, code

def process_data(df, seq_len):
    # 特征选择
    feature_cols = ['open', 'high', 'low', 'close', 'vol', 'MA5', 'MA10']
    data_raw = df[feature_cols].values
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_raw)
    
    X, y = [], []
    target_col_idx = 3 # close index
    
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i : i + seq_len])
        y.append(data_scaled[i + seq_len, target_col_idx])
        
    return np.array(X), np.array(y), scaler, target_col_idx, feature_cols

# ==========================================
# 5. 主界面逻辑
# ==========================================
st.title("📈 MambaStock 量化预测平台")
st.caption("基于 Mamba 架构的时间序列预测模型 | 支持 Tushare 数据流")

if run_btn:
    # 1. 获取数据
    with st.spinner(f"正在下载 {stock_code} 数据..."):
        start_str = start_date.strftime('%Y%m%d')
        df, full_code = get_data(ts_token, stock_code, start_str)
        
    if df is None:
        st.error(f"数据获取失败，请检查代码 {stock_code} 是否正确。")
    else:
        st.success(f"成功获取 {len(df)} 条交易数据 ({full_code})")
        
        # 2. 数据处理
        X, y, scaler, target_idx, feat_cols = process_data(df, seq_len)
        
        # 切分数据集 (最后100天做测试)
        n_test = 100
        if len(X) <= n_test:
            st.error("数据量不足以进行测试，请拉长开始日期。")
            st.stop()
            
        trainX, testX = X[:-n_test], X[-n_test:]
        trainy, testy = y[:-n_test], y[-n_test:]
        
        # 转 Tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainX_t = torch.from_numpy(trainX).float().to(device)
        trainy_t = torch.from_numpy(trainy).float().to(device)
        testX_t = torch.from_numpy(testX).float().to(device)
        
        # 3. 初始化模型
        model = Net(in_dim=len(feat_cols), out_dim=1, hidden_dim=hidden_dim, layer_num=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        
        # 4. 训练循环 (带进度条和动态图表)
        st.subheader("🛠️ 模型训练中...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty() # 用于动态画 Loss
        
        losses = []
        
        model.train()
        for e in range(epochs):
            pred = model(trainX_t)
            loss = loss_fn(pred, trainy_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # 更新前端
            if (e+1) % 5 == 0:
                progress = (e + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {e+1}/{epochs} | Loss: {loss.item():.6f}")
                
                # 绘制简单的 Loss 曲线
                chart_placeholder.line_chart(losses[-50:] if len(losses)>50 else losses)

        status_text.text("训练完成！")
        
        # 5. 预测与评估
        model.eval()
        with torch.no_grad():
            test_pred_scaled = model(testX_t).cpu().numpy()
            
        # 反归一化
        def inverse_transform_col(scaler, scaled_data, col_idx, n_features):
            dummy = np.zeros((len(scaled_data), n_features))
            dummy[:, col_idx] = scaled_data
            return scaler.inverse_transform(dummy)[:, col_idx]

        real_price = inverse_transform_col(scaler, testy, target_idx, len(feat_cols))
        pred_price = inverse_transform_col(scaler, test_pred_scaled, target_idx, len(feat_cols))
        
        # 6. 结果可视化
        st.divider()
        st.subheader("📊 预测结果分析")
        
        # 计算指标
        mse = mean_squared_error(real_price, pred_price)
        rmse = np.sqrt(mse)
        
        # 方向判断
        last_real = real_price[-1]
        last_pred = pred_price[-1]
        prev_real = real_price[-2]
        real_change = last_real - prev_real
        pred_change = last_pred - prev_real
        
        # 指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("测试集 RMSE", f"{rmse:.4f}")
        col2.metric("最新真实价", f"{last_real:.2f}", f"{real_change:.2f}")
        col3.metric("最新预测价", f"{last_pred:.2f}", f"{pred_change:.2f}")
        
        is_correct = (real_change * pred_change) > 0
        col4.metric("方向预测", "正确 ✅" if is_correct else "错误 ❌", 
                    delta_color="normal" if is_correct else "inverse")

        # 绘制交互式主图
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = df['trade_date'].iloc[-n_test:]
        
        ax.plot(dates, real_price, label='真实价格 (Real)', color='blue', linewidth=2)
        ax.plot(dates, pred_price, label='预测价格 (Predicted)', color='red', linestyle='--', linewidth=2)
        ax.set_title(f"{full_code} 股价预测对比 (Mamba Model)", fontsize=14)
        ax.set_xlabel("日期")
        ax.set_ylabel("价格")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # 原始数据展示 (可选)
        with st.expander("查看详细数据"):
            result_df = pd.DataFrame({
                "日期": dates,
                "真实价格": real_price,
                "预测价格": pred_price,
                "误差": real_price - pred_price
            })
            st.dataframe(result_df)
else:
    st.info("👈 请在左侧配置参数并点击“开始训练”")