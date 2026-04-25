import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import streamlit as st

# 加载你自己的字体文件
font = FontProperties(fname="fonts/NotoSansSC-Regular.ttf")

fig3, ax3 = plt.subplots()

# 绘图逻辑...

# 所有中文相关的地方都指定字体
ax3.set_title("周期采样信号频谱", fontproperties=font)
ax3.set_xlabel("频率(Hz)", fontproperties=font)
ax3.set_ylabel("幅值", fontproperties=font)
ax3.grid(True)
ax3.legend(loc="upper right", prop=font)  # 图例也要设置

st.pyplot(fig3)
