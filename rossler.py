import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Rossler 混沌吸引子", layout="wide")
st.title("混沌系统 - Rossler 吸引子（实时绘制）")

# 初始化状态
if "running" not in st.session_state:
    st.session_state.running = False
if "points" not in st.session_state:
    st.session_state.points = []

# 侧边栏参数
st.sidebar.header("参数设置")
a = st.sidebar.slider("a", 0.0, 1.0, 0.2, 0.01)
b = st.sidebar.slider("b", 0.0, 1.0, 0.2, 0.01)
c = st.sidebar.slider("c", 4.0, 12.0, 5.7, 0.1)
dt = 0.001  # 你原来的步长

# 按钮
col1, col2 = st.columns(2)
with col1:
    if st.button("开始绘制"):
        st.session_state.running = True
        st.session_state.x = 0.1
        st.session_state.y = 0.1
        st.session_state.z = 0.1
        st.session_state.points = []

with col2:
    if st.button("停止绘制"):
        st.session_state.running = False

plot_placeholder = st.empty()

# 核心：一次只跑一小步，避免重跑
if st.session_state.running:
    x = st.session_state.x
    y = st.session_state.y
    z = st.session_state.z
    points = st.session_state.points

    # 只迭代 N 步就刷新，这是 Streamlit 唯一稳定的实时方式
    for _ in range(50):
        # ====================== 你原来的方程，一字未改！======================
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)

        x += dx * dt
        y += dy * dt
        z += dz * dt
        # =================================================================

        points.append((x, y))

    # 保存状态
    st.session_state.x = x
    st.session_state.y = y
    st.session_state.z = z
    st.session_state.points = points

    # 绘图
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    ax.set_facecolor("black")
    ax.plot(xs, ys, color="#00ffff", lw=0.6)
    ax.axis("off")
    plot_placeholder.pyplot(fig)
    plt.close(fig)

    # 触发自动刷新，实现连续实时效果
    st.rerun()