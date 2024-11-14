import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 定义方砖块的顶点
def create_block_vertices(length, width, height):
    vertices = [
        [0, 0, 0],
        [length, 0, 0],
        [length, width, 0],
        [0, width, 0],
        [0, 0, height],
        [length, 0, height],
        [length, width, height],
        [0, width, height]
    ]
    return np.array(vertices)

# 定义方砖块的面
def create_block_faces(vertices):
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # 右面
    ]
    return faces

# 绘制方砖块
def plot_block(length, width, height):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 创建方砖块的顶点和面
    vertices = create_block_vertices(length, width, height)
    faces = create_block_faces(vertices)

    # 绘制方砖块
    poly3d = Poly3DCollection(faces, facecolors='black', edgecolor='k')
    ax.add_collection3d(poly3d)

    # 设置坐标轴范围
    ax.set_xlim([0, 2*length])
    ax.set_ylim([0, 2*width])
    ax.set_zlim([0, 2*height])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Block')

    plt.show()

if __name__ == "__main__":
    # 参数设置
    length = 10
    width = 10
    height = 5

    # 绘制方砖块
    plot_block(length, width, height)