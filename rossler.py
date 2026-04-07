import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Rossler Attractor", layout="wide")
st.title("Chaotic System - Rossler Attractor")

# Session State
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------- Tabs --------------------------
tab1, tab2 = st.tabs(["Animation", "Complexity Analysis"])

# ==================== Tab 1: Animation ====================
with tab1:
    st.subheader("Rossler Attractor Animation")

    st.sidebar.header("Parameter Settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 15.0, 5.7, 0.1, key="c")
    dt_display = st.sidebar.slider("Step Length (ms)", 0.1, 1.0, 0.3, 0.1, key="dt")
    max_steps_display = st.sidebar.slider("Max Steps (k)", 500, 5000, 2000, 500, key="steps")

    dt = dt_display / 1000
    max_steps = max_steps_display * 1000

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", key="start"):
            st.session_state.running = True
            st.session_state.history = []
            st.session_state.state = (0.1, 0.1, 0.1)

    with col2:
        if st.button("Stop", key="stop"):
            st.session_state.running = False

    placeholder = st.empty()

    if st.session_state.running:
        x, y, z = st.session_state.state
        hist = st.session_state.history

        with st.spinner("Drawing..."):
            for step in range(max_steps):
                # Rossler 方程（欧拉法，足够画出正确吸引子）
                dx = -y - z
                dy = x + a * y
                dz = b + z * (x - c)

                x += dx * dt
                y += dy * dt
                z += dz * dt

                # 只存后面一部分，避免数据太大算不动
                if step > max_steps // 2:
                    hist.append((x, y))

                if step % 2000 == 0 and not st.session_state.running:
                    break

            # 画图
            xs = [p[0] for p in hist]
            ys = [p[1] for p in hist]

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_facecolor("black")
            ax.plot(xs, ys, color="#00ffff", linewidth=0.5)
            ax.axis("off")
            placeholder.pyplot(fig)
            plt.close(fig)

            st.session_state.state = (x, y, z)
            st.session_state.history = hist
            st.session_state.running = False
            st.success("Drawing completed!")

# ==================== Tab 2: Complexity Analysis ====================
with tab2:
    st.subheader("Chaos Complexity Analysis")

    # 轻量 0-1 混沌测试（稳定不崩）
    def chaos_01_simple(series):
        try:
            N = len(series)
            t = np.arange(N)
            c = np.pi / 3
            p = np.cumsum(series * np.cos(c * t))
            q = np.cumsum(series * np.sin(c * t))
            log_t = np.log(t[1:]+1e-9)
            log_M = np.log(np.sqrt(p[1:]**2 + q[1:]**2)+1e-9)
            k = np.polyfit(log_t, log_M, 1)[0]
            return round(k, 4)
        except:
            return 0.0

    # 简易李雅普诺夫指数（稳定、不NaN）
    def simple_lyapunov(series, dt):
        try:
            d = np.abs(np.diff(series))
            d = d[d > 1e-8]
            le = np.mean(np.log(d)) / dt
            return round(le, 4)
        except:
            return -999

    # 相空间重构
    def phase_reconstruct(series, tau=15):
        n = len(series) - tau
        return np.array([[series[i], series[i+tau]] for i in range(n)])

    # ================== 计算按钮 ==================
    if st.button("Compute Complexity Indicators"):
        hist = st.session_state.history
        if len(hist) < 1000:
            st.warning("Please run Animation first with more steps!")
        else:
            with st.spinner("Calculating..."):
                xs = np.array([p[0] for p in hist])
                dt = 0.0003

                # 1. 李雅普诺夫指数
                lyap = simple_lyapunov(xs, dt)
                is_chaotic = lyap > 0.02

                # 2. 0-1 测试
                K = chaos_01_simple(xs)
                k_chaos = K > 0.5

                # 3. 相空间重构
                psr = phase_reconstruct(xs, tau=15)

                # 4. 功率谱
                f, Pxx = welch(xs, fs=1/dt, nperseg=512)

            # ================== 显示结果 ==================
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

            # 画图
            st.markdown("### Power Spectrum & Phase Space")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(f, Pxx, color='c')
            ax1.set_title("Power Spectrum")
            ax1.grid(alpha=0.3)

            ax2.plot(psr[:,0], psr[:,1], color='m', linewidth=0.4)
            ax2.set_title("Phase Space Reconstruction")
            ax2.axis('equal')
            st.pyplot(fig)
            plt.close(fig)

            st.success("Calculation finished!")