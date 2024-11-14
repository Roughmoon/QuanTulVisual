from sympy import *
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# 定义一个方形势垒
def SqBarrier(width, V0, X):  # 依次为：势垒宽度、势垒高度、网格点向量
    V = np.array([V0 if 0.0 < x < width else 0.0 for x in X])  # list to array
    return V

# 定义一个一维一般势垒
def GeBarrier(width, func: callable, Xv):  # 依次为：势垒宽度、接收的用户定义函数、网格点向量
    """
    通过一个给定的数学函数，对每一个网格点X中的点，求解对应函数值，赋给一个新的向量。
    """
    V = np.array([func(Xv[i]) if 0.0 <= Xv[i] <= width else 0.0 for i in range(0, len(Xv))])  # First number is Zero=0
    return V

# 尝试定义一个平面波
def Plw(x, x0, sigma):  # 依次为：网格点向量、高斯波包中心位置、离散度
    return np.cos((-(x - x0)) / (4 * sigma ** 2)) / (2 * np.pi * sigma ** 2) ** (1 / 4)

# 定义一个高斯函数（高斯波包，不包含虚数部分，非完整高斯波包），已归一化
def NorGaussLt(x, x0, sigma):  # 依次为：网格点向量、高斯波包中心位置、离散度
    return np.exp((-(x - x0) ** 2) / (4 * sigma ** 2)) / (2 * np.pi * sigma ** 2) ** (1 / 4)

class WaveSet:
    def __init__(self, V, N, k0, Xv, dt=0.4, m=1.0, hbar=1.0, sigma=40.0):
        # 依次为：对象、接收的势垒向量、网格点数量、波矢、网格点向量、时间差、质量、hbar、离散度。
        self.cc = 0
        self.N = N
        self.X = Xv
        self.dx = (Xv[1] - Xv[0])
        self.sigma = sigma
        self.m = m
        self.dt = dt
        self.x0 = -(round(self.N / 2) - 15 * sigma)  # 函数中心位置
        self.V = V  # 这个V是个向量，共有N个数据
        self.hbar = hbar
        self.k0 = k0
        self.psi_r = np.zeros((3, N))
        self.psi_i = np.zeros((3, N))
        self.psi_p = np.zeros(N, )
        self.I1 = range(1, N - 1)
        self.I2 = range(2, N - 0)
        self.I3 = range(0, N - 2)

    def init(self):  # initialization
        wave = NorGaussLt(self.X, self.x0, self.sigma)
        # 高斯波包初始化
        self.psi_r[0, :] = np.cos(self.k0 * self.X) * wave
        self.psi_r[1, :] = np.cos(self.k0 * self.X) * wave
        self.psi_i[0, :] = np.sin(self.k0 * self.X) * wave
        self.psi_i[1, :] = np.sin(self.k0 * self.X) * wave

    def develop(self, num):
        A = self.hbar * self.dt / (self.m * self.dx ** 2)  # e23
        B = 2 * self.dt / self.hbar
        Cv = B * self.V
        for i in range(num):
            self.psi_i[2, self.I1] = self.psi_i[0, self.I1] + A * (self.psi_r[1][self.I2] - 2 * self.psi_r[1][self.I1] + self.psi_r[1][self.I3])
            self.psi_i[2] -= Cv * self.psi_r[1]
            self.psi_r[2, self.I1] = self.psi_r[0, self.I1] - A * (self.psi_i[1][self.I2] - 2 * self.psi_i[1][self.I1] + self.psi_i[1][self.I3])
            self.psi_r[2] += Cv * self.psi_i[1]

            self.psi_r[0] = self.psi_r[1]
            self.psi_r[1] = self.psi_r[2]
            self.psi_i[0] = self.psi_i[1]
            self.psi_i[1] = self.psi_i[2]

        self.psi_p = np.sqrt(self.psi_r[1] ** 2 + self.psi_i[1] ** 2)

        return self.psi_p
# 定义方砖块的顶点
def create_block_vertices(x, y, z, dx, dy, dz):
    return np.array([
        [x - dx/2, y - dy/2, z - dz/2],          # P0
        [x + dx/2, y - dy/2, z - dz/2],          # P1
        [x + dx/2, y + dy/2, z - dz/2],           # P2
        [x - dx/2, y + dy/2, z - dz/2],           # P3
        [x - dx/2, y - dy/2, z + dz/2],           # P4
        [x + dx/2, y - dy/2, z + dz/2],           # P5
        [x + dx/2, y + dy/2, z + dz/2],           # P6
        [x - dx/2, y + dy/2, z + dz/2]            # P7
    ])

# 定义方砖块的面
def create_block_faces(vertices):
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # 右面
    ]

class Animator:
    def __init__(self, waveset):
        self.count = 0
        self.waveset = waveset  # 传递对象
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(self.waveset.X.min(), self.waveset.X.max())
        self.ax.set_ylim(-1.5 * ((self.waveset.psi_r[1]).max()), 1.5 * ((self.waveset.psi_r[1]).max()))
        self.ax.set_zlim(-1.5 * ((self.waveset.psi_r[1]).max()), 1.5 * ((self.waveset.psi_r[1]).max()))

        # 初始化线条对象
        self.realPart, = self.ax.plot([], [], [], c="r", alpha=0.5, label=r"$\psi(x)_{real}$", linewidth=1)
        self.imaPart, = self.ax.plot([], [], [], c="b", alpha=0.5, label=r"$\psi(x)_{imaginary}$", linewidth=1)
        self.title = self.ax.set_title("")
        self.ax.legend(prop=dict(size=5))
        self.ax.set_xlabel("$x$")

        # 清除旧的砖块
        self.blocks = []
        # 绘制势垒
        dx = 0
        for i in range(len(self.waveset.V)):
            if self.waveset.V[i] > 0:
                x = self.waveset.X[i]
                y = 0
                z = 0
                dx = +0.5
                dy = self.waveset.V[i]/2  # 假设势垒的宽度固定为1
                dz = self.waveset.V[i]/2

                vertices = create_block_vertices(x, y, z, dx, dy, dz)
                faces = create_block_faces(vertices)

                block = Poly3DCollection(faces, facecolors='black', alpha=1)
                self.ax.add_collection3d(block)
                self.blocks.append(block)
        
        
        
        # 绘制势垒
        # self.ax.plot(self.waveset.X, np.zeros_like(self.waveset.X), self.waveset.V, c="k", alpha=1, linewidth=2)

    def init(self):
        self.realPart.set_data([], [])
        self.realPart.set_3d_properties([])
        self.imaPart.set_data([], [])
        self.imaPart.set_3d_properties([])
        self.title.set_text("")
        return (self.realPart, self.imaPart)

    def update(self, num):
        self.waveset.develop(10)
        self.realPart.set_data(self.waveset.X, self.waveset.psi_r[1])
        self.realPart.set_3d_properties(np.zeros_like(self.waveset.psi_r[1]))
        self.imaPart.set_data(self.waveset.X, np.zeros_like(self.waveset.psi_i[1]))
        self.imaPart.set_3d_properties(self.waveset.psi_i[1])
        self.title.set_text(f"Time Step: {num}")
        self.count += 1
        return (self.realPart, self.imaPart)

    def time_step(self):
        while True:
            if self.count == 377:
                self.waveset.init()
                self.count = 0
            yield self.count

    def animate(self, save=False, save_images=True, image_dir=None):
        self.ani = animation.FuncAnimation(self.fig, self.update, self.time_step, init_func=self.init, interval=20, save_count=299)
        plt.show()

if __name__ == "__main__":
    N = 601
    x = symbols('x')
    func = 9e-2
    func = lambdify(x, sympify(func))
    width = 5
    xlim = 300
    V = GeBarrier(width=width, func=func, Xv=np.linspace(-xlim, xlim, N))
    waveset = WaveSet(V=V, N=N, k0=(np.pi) / 20, Xv=np.linspace(-xlim, xlim, N), m=1, hbar=1, sigma=10)
    waveset.init()
    animator = Animator(waveset)
    animator.animate(save=False, save_images=False)