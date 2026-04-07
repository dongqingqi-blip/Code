import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.spatial import KDTree

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

# ==================== Tab 1: Animation ====================
with tab1:
    st.subheader("Rossler Attractor Animation")

    st.sidebar.header("Parameter Settings")
    a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01, key="a")
    b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01, key="b")
    c = st.sidebar.slider("c", 0.0, 10.0, 5.7, 0.1, key="c")
    dt_display = st.sidebar.slider("Step Length (ms)", 0.1, 1.0, 0.5, 0.1, key="dt")
    max_steps_display = st.sidebar.slider("Max Steps (k)", 100, 5000, 1000, 100, key="steps")

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
                # RK4 积分
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

                # 修复：存储完整x,y,z轨迹
                hist.append((x, y, z))

                if step % 500 == 0 and not st.session_state.running:
                    break

            xs = [p[0] for p in hist]
            ys = [p[1] for p in hist]

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_facecolor("black")
            ax.plot(xs, ys, color="#00ffff", linewidth=0.6)
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

    def correlation_dimension(series, k=5):
        series = series.reshape(-1, 1)
        tree = KDTree(series)
        dists, _ = tree.query(series, k=k+1)
        dists = dists[:, 1:]
        dists = dists[dists > 1e-9]
        log_eps = np.log(np.sort(dists))
        log_C = np.log(np.arange(1, len(log_eps)+1) / len(log_eps))
        fit = np.polyfit(log_eps[:len(log_eps)//2], log_C[:len(log_eps)//2], 1)
        return fit[0]

    def phase_space_reconstruction(series, tau=1, m=2):
        N = len(series)
        return np.array([series[i:i+m*tau:tau] for i in range(N - m*tau + 1)])

    def chaos_01_test(series):
        N = len(series)
        n = np.arange(N)
        p, q = np.zeros(N), np.zeros(N)
        c = np.pi / 4
        for i in range(1, N):
            p[i] = p[i-1] + series[i-1] * np.cos(c*i)
            q[i] = q[i-1] + series[i-1] * np.sin(c*i)
        M = np.sqrt(p**2 + q**2)
        K = np.polyfit(np.log(n[1:]), np.log(M[1:]), 1)[0]
        return K


    def correct_lyapunov(x_traj, y_traj, z_traj, dt, a, b, c):
        n = len(x_traj)
        log_divergence = []
        dx0 = 1e-6
        # 初始扰动
        x1 = x_traj.copy()
        y1 = y_traj.copy()
        z1 = z_traj.copy()
        x2 = x1 + dx0
        y2 = y1.copy()
        z2 = z1.copy()

        # 仅定义一次Rossler方程，避免循环内重复定义
        def rossler(x, y, z):
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            return dx, dy, dz

        for i in range(n - 1):
            # ---------------- 轨道1：完整RK4积分 ----------------
            k1x1, k1y1, k1z1 = rossler(x1[i], y1[i], z1[i])
            k2x1, k2y1, k2z1 = rossler(x1[i] + dt * k1x1 / 2, y1[i] + dt * k1y1 / 2, z1[i] + dt * k1z1 / 2)
            k3x1, k3y1, k3z1 = rossler(x1[i] + dt * k2x1 / 2, y1[i] + dt * k2y1 / 2, z1[i] + dt * k2z1 / 2)
            k4x1, k4y1, k4z1 = rossler(x1[i] + dt * k3x1, y1[i] + dt * k3y1, z1[i] + dt * k3z1)

            x1[i + 1] = x1[i] + dt / 6 * (k1x1 + 2 * k2x1 + 2 * k3x1 + k4x1)
            y1[i + 1] = y1[i] + dt / 6 * (k1y1 + 2 * k2y1 + 2 * k3y1 + k4y1)
            z1[i + 1] = z1[i] + dt / 6 * (k1z1 + 2 * k2z1 + 2 * k3z1 + k4z1)

            # ---------------- 轨道2：完整RK4积分 ----------------
            k1x2, k1y2, k1z2 = rossler(x2[i], y2[i], z2[i])
            k2x2, k2y2, k2z2 = rossler(x2[i] + dt * k1x2 / 2, y2[i] + dt * k1y2 / 2, z2[i] + dt * k1z2 / 2)
            k3x2, k3y2, k3z2 = rossler(x2[i] + dt * k2x2 / 2, y2[i] + dt * k2y2 / 2, z2[i] + dt * k2z2 / 2)
            k4x2, k4y2, k4z2 = rossler(x2[i] + dt * k3x2, y2[i] + dt * k3y2, z2[i] + dt * k3z2)

            x2[i + 1] = x2[i] + dt / 6 * (k1x2 + 2 * k2x2 + 2 * k3x2 + k4x2)
            y2[i + 1] = y2[i] + dt / 6 * (k1y2 + 2 * k2y2 + 2 * k3y2 + k4y2)
            z2[i + 1] = z2[i] + dt / 6 * (k1z2 + 2 * k2z2 + 2 * k3z2 + k4z2)

            # 计算距离并归一化扰动（保持扰动大小为dx0）
            d = np.sqrt((x2[i + 1] - x1[i + 1]) ** 2 + (y2[i + 1] - y1[i + 1]) ** 2 + (z2[i + 1] - z1[i + 1]) ** 2)
            d = max(d, 1e-12)  # 避免除零
            x2[i + 1] = x1[i + 1] + dx0 * (x2[i + 1] - x1[i + 1]) / d
            y2[i + 1] = y1[i + 1] + dx0 * (y2[i + 1] - y1[i + 1]) / d
            z2[i + 1] = z1[i + 1] + dx0 * (z2[i + 1] - z1[i + 1]) / d

            log_divergence.append(np.log(d / dx0))

        return np.mean(log_divergence) / dt

    if st.button("Compute Complexity Indicators", key="compute"):
        hist = st.session_state.history
        if len(hist) < 200:
            st.warning("Please run the animation first to generate trajectory data!")
        else:
            with st.spinner("Calculating..."):
                traj = np.array(hist)
                xs = traj[:,0]
                ys = traj[:,1]
                zs = traj[:,2]

                # 修复：调用参数匹配，使用实时参数
                lyap = correct_lyapunov(xs, ys, zs, dt, a, b, c)
                lyap = round(lyap, 4)
                is_chaotic = lyap > 0.01

                corr_dim = correlation_dimension(xs)
                corr_dim = round(corr_dim, 4)

                K = chaos_01_test(xs)
                K = round(K, 4)
                k_chaos = K > 0.5

                f, Pxx = welch(xs, fs=1000, nperseg=1024)
                psr = phase_space_reconstruction(xs, tau=1, m=2)

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

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(f, Pxx, color='c')
                ax1.set_title("Power Spectrum")
                ax1.grid(alpha=0.3)

                ax2.plot(psr[:,0], psr[:,1], color='magenta', linewidth=0.5)
                ax2.set_title("Phase Space Reconstruction")
                ax2.axis('equal')
                st.pyplot(fig)
                plt.close(fig)

            st.success("All complexity indicators computed successfully!")