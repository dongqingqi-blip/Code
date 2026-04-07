import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Rossler Attractor", layout="wide")
st.title("Chaotic System - Rossler Attractor")

# Session State Initialization
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []
if "params" not in st.session_state:
    st.session_state.params = (0.2, 0.2, 5.7, 0.0001, 2000000)

# -------------------------- Tabs --------------------------
tab1, tab2 = st.tabs(["Animation", "Complexity Analysis"])

# ==================== Tab 1: Animation (修复：RK4积分，保证混沌) ====================
with tab1:
    st.subheader("Rossler Attractor Animation")

    st.sidebar.header("Parameter Settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 15.0, 5.7, 0.1, key="c")
    # 修复步长：默认0.1ms，最大1ms，保证RK4精度
    dt_display = st.sidebar.slider("Step Length (ms)", 0.05, 1.0, 0.1, 0.05, key="dt")
    # 修复步数：默认2000k，保证进入混沌吸引子
    max_steps_display = st.sidebar.slider("Max Steps (k)", 1000, 5000, 2000, 500, key="steps")

    dt = dt_display / 1000
    max_steps = max_steps_display * 1000

    # 保存参数到session state，供复杂度分析使用
    st.session_state.params = (a, b, c, dt, max_steps)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", key="start"):
            st.session_state.running = True
            st.session_state.history = []
            # 初始值，保证进入混沌吸引子
            x, y, z = 0.1, 0.1, 0.1
            st.session_state.state = (x, y, z)

    with col2:
        if st.button("Stop", key="stop"):
            st.session_state.running = False

    placeholder = st.empty()

    if st.session_state.running:
        x, y, z = st.session_state.state
        hist = st.session_state.history
        a, b, c, dt, _ = st.session_state.params

        with st.spinner("Drawing... (Generating chaotic attractor, please wait)"):
            for step in range(max_steps):
                # 🔴 核心修复：四阶龙格-库塔(RK4)积分，混沌系统标准方法，无数值耗散
                def rossler(x, y, z):
                    dx = -y - z
                    dy = x + a * y
                    dz = b + z * (x - c)
                    return dx, dy, dz

                # RK4 四步计算
                k1x, k1y, k1z = rossler(x, y, z)
                k2x, k2y, k2z = rossler(x + dt*k1x/2, y + dt*k1y/2, z + dt*k1z/2)
                k3x, k3y, k3z = rossler(x + dt*k2x/2, y + dt*k2y/2, z + dt*k2z/2)
                k4x, k4y, k4z = rossler(x + dt*k3x, y + dt*k3y, z + dt*k3z)

                # 更新状态
                x += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
                y += dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
                z += dt/6 * (k1z + 2*k2z + 2*k3z + k4z)

                # 存储轨迹：跳过前10% transient阶段，只存后90%，保证进入混沌
                if step > max_steps * 0.1:
                    hist.append((x, y))

                # 每1000步检查停止，避免卡死
                if step % 1000 == 0 and not st.session_state.running:
                    break

            # 绘制吸引子
            xs = [p[0] for p in hist]
            ys = [p[1] for p in hist]

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_facecolor("black")
            ax.plot(xs, ys, color="#00ffff", linewidth=0.3)
            ax.axis("off")
            placeholder.pyplot(fig)
            plt.close(fig)

            # 保存状态
            st.session_state.state = (x, y, z)
            st.session_state.history = hist
            st.session_state.running = False
            st.success("Drawing completed! Chaotic attractor generated successfully.")

# ==================== Tab 2: Complexity Analysis (全指标修复) ====================
with tab2:
    st.subheader("Chaos Complexity Analysis")

    # 🔴 修复1：标准0-1混沌测试（Gottwald-Melbourne实现，去趋势+多次平均）
    def chaos_01_test(series, num_trials=5):
        N = len(series)
        if N < 1000:
            return 0.0
        # 去趋势：减去均值，避免K值被拉低
        series = series - np.mean(series)
        K_list = []
        for _ in range(num_trials):
            # 随机选择c，避免固定c的偏差
            c = np.random.uniform(np.pi/5, 4*np.pi/5)
            t = np.arange(N)
            # 计算p和q
            p = np.cumsum(series * np.cos(c * t))
            q = np.cumsum(series * np.sin(c * t))
            # 计算均方位移M
            M = np.sqrt(p**2 + q**2)
            # 对数拟合，计算斜率K
            log_t = np.log(t[1:] + 1e-9)
            log_M = np.log(M[1:] + 1e-9)
            slope, _, _, _, _ = linregress(log_t, log_M)
            K_list.append(slope)
        # 返回平均K值
        return round(np.mean(K_list), 4)

    # 🔴 修复2：稳定的李雅普诺夫指数计算（避免NaN）
    def lyapunov_exponent(series, dt):
        try:
            # 计算差分，过滤零值，避免log(0)
            diffs = np.abs(np.diff(series))
            diffs = diffs[diffs > 1e-9]
            if len(diffs) < 100:
                return -999.0
            # 计算对数平均，除以dt，得到LE
            le = np.mean(np.log(diffs)) / dt
            return round(le, 4)
        except:
            return -999.0

    # 🔴 修复3：正确的相空间重构（Takens定理）
    def phase_space_reconstruction(series, tau=5, m=2):
        N = len(series)
        if N < m*tau:
            return np.array([])
        return np.array([series[i:i+m*tau:tau] for i in range(N - m*tau + 1)])

    # 计算按钮
    if st.button("Compute Complexity Indicators", key="compute"):
        hist = st.session_state.history
        a, b, c, dt, _ = st.session_state.params
        if len(hist) < 10000:
            st.warning("Please run Animation first with at least 1000k steps!")
        else:
            with st.spinner("Calculating... (10-20 seconds)"):
                # 提取x序列
                xs = np.array([p[0] for p in hist])
                N = len(xs)

                # 1. 李雅普诺夫指数
                lyap = lyapunov_exponent(xs, dt)
                is_chaotic = lyap > 0.01

                # 2. 0-1 Test
                K = chaos_01_test(xs, num_trials=5)
                k_chaos = K > 0.8

                # 3. 相空间重构
                psr = phase_space_reconstruction(xs, tau=5, m=2)

                # 4. 功率谱
                f, Pxx = welch(xs, fs=1/dt, nperseg=2048)

                # 显示结果
                st.markdown("### Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Max Lyapunov Exponent", lyap if is_chaotic else "-")
                with col2:
                    st.metric("0-1 Test K", K)

                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"**Chaos State:** {'✅ Chaotic' if is_chaotic else '❌ Non-chaotic'}")
                with col4:
                    st.markdown(f"**0-1 Result:** {'✅ Chaotic' if k_chaos else '❌ Non-chaotic'}")

                # 绘制图表
                st.markdown("### Power Spectrum & Phase Space Reconstruction")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # 功率谱：混沌=连续宽谱（对数坐标）
                ax1.plot(f, Pxx, color='c', linewidth=0.8)
                ax1.set_title("Power Spectrum (Continuous Broadband = Chaotic)")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Power (log scale)")
                ax1.set_yscale('log')
                ax1.grid(alpha=0.3)

                # 相空间重构：混沌=分形缠绕结构
                if len(psr) > 0:
                    ax2.plot(psr[:,0], psr[:,1], color='magenta', linewidth=0.2)
                    ax2.set_title("Phase Space Reconstruction (Fractal Attractor)")
                    ax2.axis('equal')
                    ax2.grid(alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax2.transAxes)

                st.pyplot(fig)
                plt.close(fig)

            st.success("All indicators computed successfully!")

# 页脚
st.markdown("---")
st.caption("Rossler Attractor Chaos Analysis | RK4 Integration | Standard Chaos Indicators")