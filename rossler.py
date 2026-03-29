import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Rossler 混沌吸引子", layout="wide")
st.title("混沌系统 - Rossler 吸引子")

# 控制状态
if "running" not in st.session_state:
    st.session_state.running = False

# 侧边栏参数
st.sidebar.header("Parameter Settings")
a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01)
b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01)
c = st.sidebar.slider("c", 4.0, 12.0, 5.7, 0.1)
dt = st.sidebar.slider("step length", 0.001, 0.01, 0.001, 0.001)

# 按钮
col1, col2 = st.columns(2)
with col1:
    if st.button("Start"):
        st.session_state.running = True
        st.session_state.x = 0.1
        st.session_state.y = 0.1
        st.session_state.z = 0.1
        st.session_state.points = []

with col2:
    if st.button("Stop"):
        st.session_state.running = False

# 绘图占位符
plot_placeholder = st.empty()

# 实时绘制主循环
if st.session_state.running:
    x = st.session_state.x
    y = st.session_state.y
    z = st.session_state.z
    points = st.session_state.points

    # 无限实时绘制，直到停止
    while st.session_state.running:
        # 你的原始 Rossler 方程，完全没改
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)

        x += dx * dt
        y += dy * dt
        z += dz * dt

        points.append((x, y))

        # 每一步都实时刷新！！！
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
        ax.set_facecolor("black")
        ax.plot(xs, ys, color="#00ffff", linewidth=0.7)
        ax.axis("off")
        plot_placeholder.pyplot(fig)
        plt.close(fig)

        # 保存状态
        st.session_state.x = x
        st.session_state.y = y
        st.session_state.z = z
        st.session_state.points = points