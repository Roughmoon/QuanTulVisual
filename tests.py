import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# 创建数据
t = np.linspace(0, 2 * np.pi, 100)  # 时间轴
x = np.sin(t)  # 第一条波动线的 x 坐标
y = np.sin(t)  # 第二条波动线的 y 坐标
z = t          # 两条波动线的 z 坐标

# 设置初始数据
x_data, y_data, z_data = [], [], []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义更新函数
def update(num):
    ax.cla()  # 清除之前的图像
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # 更新数据
    x_data.append(z[num])
    y_data.append(np.sin(t[num]))
    z_data.append(np.sin(t[num]))

    # 绘制 X-Z 平面的波动线
    ax.plot(z[:num], x[:num], np.zeros_like(x[:num]), label='X-Z Wave', color='r')
    
    # 绘制 Y-Z 平面的波动线
    ax.plot(z[:num], np.zeros_like(y[:num]), y[:num], label='Y-Z Wave', color='b')

    ax.legend()

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)

plt.show()