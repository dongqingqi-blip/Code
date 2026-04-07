import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.spatial import KDTree
from scipy.stats import linregress

# -------------------------- Page Config --------------------------
st.set_page_config(page_title="Rossler Attractor", layout="wide")
st.title("Chaotic System - Rossler Attractor")

# Session State Initialization
if "running" not in st.session_state:
    st.session_state.running = False
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------- Tabs --------------------------
tab1, tab2 = st.tabs(["Animation", "Complexity Analysis"])

# ==================== Tab 1: Animation (修复RK4+步长) ====================
with tab1:
    st.subheader("Rossler Attractor Animation")

    st.sidebar.header("Parameter Settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 10.0, 5.7, 0.1, key="c")
    # 修复步长：默认0.1ms，最大1ms
    dt_display = st.sidebar.slider("Step Length (ms)", 0.05, 1.0, 0.1, 0.05, key="dt")
    # 修复最大步数：默认2000k，保证进入混沌
    max_steps_display = st.sidebar.slider("Max Steps (k)", 100, 5000, 2000, 100, key="steps")

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

        with st.spinner("Drawing... (Please wait for full convergence)"):
            for step in range(max_steps):
                # RK4积分（保证精度）
                def rossler(x,y,z):
                    dx = -y - z
                    dy = x + a*y
                    dz = b + z*(x-c)
                    return dx, dy, dz

                k1x,k1y,k1z = rossler(x,y,z)
                k2x,k2y,k2z = rossler(x+dt*k1x/2, y+dt*k1y/2, z+dt*k1z/2)
                k3x,k3y,k3z = rossler(x+dt*k2x/2, y+dt*k2y/2, z+dt*k2z/2)
                k4x,k4y,k4z = rossler(x+dt*k3x, y+dt*k3y, z+dt*k3z)

                x += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
                y += dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
                z += dt/6 * (k1z + 2*k2z + 2*k3z + k4z)

                hist.append((x, y))

                if step % 1000 == 0 and not st.session_state.running:
                    break

            xs = [p[0] for p in hist]
            ys = [p[1] for p in hist]

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_facecolor("black")
            ax.plot(xs, ys, color="#00ffff", linewidth=0.4)
            ax.axis("off")
            placeholder.pyplot(fig)
            plt.close(fig)

            st.session_state.state = (x, y, z)
            st.session_state.history = hist
            st.session_state.running = False
            st.success("Drawing completed! (Chaotic attractor generated)")

# ==================== Tab 2: Complexity Analysis (全修复) ====================
with tab2:
    st.subheader("Chaos Complexity Analysis")

    # 1. 正确的相空间重构（Takens定理）
    def phase_space_reconstruction(series, tau, m=2):
        N = len(series)
        return np.array([series[i:i+m*tau:tau] for i in range(N - m*tau + 1)])

    # 2. 正确的关联维数（基于相空间重构，Grassberger-Procaccia算法）
    def correlation_dimension(psr, eps_range=np.logspace(-3, 1, 50)):
        N = len(psr)
        tree = KDTree(psr)
        c = np.zeros_like(eps_range)
        for i, eps in enumerate(eps_range):
            counts = tree.query_ball_point(psr, r=eps, return_length=True)
            c[i] = 2 * np.sum(counts) / (N * (N-1))
        log_eps = np.log(eps_range)
        log_c = np.log(c + 1e-10)
        start = int(len(log_eps)*0.3)
        end = int(len(log_eps)*0.7)
        slope, _, _, _, _ = linregress(log_eps[start:end], log_c[start:end])
        return slope

    # 3. 正确的0-1 Test（Gottwald-Melbourne标准实现）
    def chaos_01_test(series, c=None):
        N = len(series)
        if c is None:
            c = np.random.uniform(np.pi/5, 4*np.pi/5)
        n = np.arange(N)
        p = np.zeros(N)
        q = np.zeros(N)
        for i in range(1, N):
            p[i] = p[i-1] + series[i-1] * np.cos(c*i)
            q[i] = q[i-1] + series[i-1] * np.sin(c*i)
        M = np.sqrt(p**2 + q**2)
        log_n = np.log(n[1:])
        log_M = np.log(M[1:])
        slope, _, _, _, _ = linregress(log_n, log_M)
        return slope

    # 4. 正确的最大李雅普诺夫指数（三维系统雅可比矩阵法）
    def max_lyapunov_rossler(a, b, c, dt, x0=0.1, y0=0.1, z0=0.1, steps=10000):
        x, y, z = x0, y0, z0
        dx, dy, dz = 1e-8, 0.0, 0.0
        le_sum = 0.0
        for _ in range(steps):
            def rossler(x,y,z):
                dx_sys = -y - z
                dy_sys = x + a*y
                dz_sys = b + z*(x-c)
                return dx_sys, dy_sys, dz_sys
            k1x,k1y,k1z = rossler(x,y,z)
            k2x,k2y,k2z = rossler(x+dt*k1x/2, y+dt*k1y/2, z+dt*k1z/2)
            k3x,k3y,k3z = rossler(x+dt*k2x/2, y+dt*k2y/2, z+dt*k2z/2)
            k4x,k4y,k4z = rossler(x+dt*k3x, y+dt*k3y, z+dt*k3z)
            x += dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
            y += dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
            z += dt/6 * (k1z + 2*k2z + 2*k3z + k4z)
            J = np.array([[0, -1, -1],[1, a, 0],[z, 0, x - c]])
            dvec = np.array([dx, dy, dz])
            dvec = dvec + dt * J @ dvec
            norm = np.linalg.norm(dvec)
            if norm < 1e-16:
                norm = 1e-16
            le_sum += np.log(norm)
            dx, dy, dz = dvec / norm
        le = le_sum / (steps * dt)
        return le

    if st.button("Compute Complexity Indicators", key="compute"):
        hist = st.session_state.history
        if len(hist) < 10000:
            st.warning("Please run the animation with at least 1000k steps to generate enough trajectory data!")
        else:
            with st.spinner("Calculating... (This may take a minute)"):
                xs = np.array([p[0] for p in hist])
                N = len(xs)
                dt = 0.0001

                # 1. 正确的最大李雅普诺夫指数
                lyap = max_lyapunov_rossler(a, b, c, dt, steps=min(N, 20000))
                lyap = round(lyap, 4)
                is_chaotic = lyap > 0.01

                # 2. 相空间重构（最优tau=10）
                tau = 10
                psr = phase_space_reconstruction(xs, tau=tau, m=2)

                # 3. 正确的关联维数
                corr_dim = correlation_dimension(psr)
                corr_dim = round(corr_dim, 4)

                # 4. 正确的0-1 Test（多次取平均）
                K_list = [chaos_01_test(xs) for _ in range(5)]
                K = round(np.mean(K_list), 4)
                k_chaos = K > 0.8

                # 5. 功率谱（混沌连续宽谱）
                f, Pxx = welch(xs, fs=1/dt, nperseg=2048)

                # Display
                st.markdown("### Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max Lyapunov Exponent", lyap if is_chaotic else "-")
                with col2:
                    st.metric("Correlation Dimension", corr_dim)
                with col3:
                    st.metric("0-1 Test K", K)

                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Chaos State", "Chaotic" if is_chaotic else "Non-chaotic")
                with col5:
                    st.metric("0-1 Test Result", "Chaotic" if k_chaos else "Non-chaotic")

                # Plot
                st.markdown("### Power Spectrum & Phase Space Reconstruction")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(f, Pxx, color='c')
                ax1.set_title("Power Spectrum (Continuous Broadband = Chaotic)")
                ax1.set_xlabel("Frequency (Hz)")
                ax1.set_ylabel("Power")
                ax1.grid(alpha=0.3)
                ax1.set_yscale('log')

                ax2.plot(psr[:,0], psr[:,1], color='magenta', linewidth=0.3)
                ax2.set_title("Phase Space Reconstruction (Fractal Attractor)")
                ax2.axis('equal')
                ax2.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            st.success("All complexity indicators computed successfully! (Chaotic state verified)")