from sympy import *
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
import os

#定义一个方形势垒
def SqBarrier(width,V0,X):#依次为：势垒宽度、势垒高度、网格点向量
    V = np.array([V0 if 0.0 < x < width else 0.0 for x in X])#list to array
    return V

#定义一个一维一般势垒
def GeBarrier(width,func:callable,Xv):#依次为：势垒宽度、接收的用户定义函数、网格点向量
    """
    通过一个给定的数学函数，对每一个网格点X中的点，求解对应函数值，赋给一个新的向量。
    """
    V = np.array([func(Xv[i]) if 0.0 <= Xv[i] <= width else 0.0 for i in range(0,len(Xv))])#First number is Zero=0
    return V

#尝试定义一个平面波
def Plw(x,x0,sigma):#依次为：网格点向量、高斯波包中心位置、离散度
    return np.cos((-(x-x0))/(4*sigma**2))/(2*pi*sigma**2)**(1/4)
#定义一个高斯函数（高斯波包，不包含虚数部分，非完整高斯波包），已归一化
def NorGaussLt(x,x0,sigma):#依次为：网格点向量、高斯波包中心位置、离散度
    return np.exp((-(x-x0)**2)/(4*sigma**2))/(2*pi*sigma**2)**(1/4)
class Calculation:
    #A_1,A_2,B_1,B_2,C_1,C_2分别为方形势垒的左、中、右区域的波函数的解的系数
    A_1 , A_2 , B_1 , B_2 , C_1 , C_2 = symbols('A_1 , A_2 , B_1 , B_2 , C_1 , C_2 ')
    # A_01,A_02,B_01,B_02,C_01,C_02分别为U==E条件下的对应内容
    A_01 , A_02 , B_01 , B_02 , C_01 , C_02 = symbols('A_01 , A_02 , B_01 , B_02 , C_01 , C_02 ')
    #k_1=sqrt(2*m*E/h_bar**2),k_2=sqrt(2*m*(E-U_0)/h_bar**2)
    k_1 , k_2 = symbols('k_1 , k_2')
    #D为透射系数，D=abs(C_1)**2/abs(A_1)**2，R为反射系数，R=1-D，D0和R0分别是U==E条件下的透射系数和反射系数
    D , R , D0 , R0 = symbols('D , R , D0 , R0')
    #a为势垒宽度
    a = symbols('a')
    x = symbols('x')
    solutions = {}
    solutions0 = {}
    #接下来的符号为计算D和R做准备,其中：m为电子质量，E为粒子能量，U为方形势垒的势能
    #为方便计算，将h_bar记为h0，并且h0=1.05e-34，m=9.11e-31

    def CalcuDR(self,):
        self.solutions = solve([self.A_1+self.A_2-self.B_1-self.B_2,
                        self.k_1*self.A_1-self.k_1*self.A_2-self.k_2*self.B_1+self.k_2*self.B_2,
                        self.B_1*exp(I*self.k_2*self.a)+self.B_2*exp(-I*self.k_2*self.a)-self.C_1*exp(I*self.k_1*self.a),
                        self.k_2*self.B_1*exp(I*self.k_2*self.a)-self.k_2*self.B_2*exp(-I*self.k_2*self.a)-self.k_1*self.C_1*exp(I*self.k_1*self.a)],
                        [self.C_1,self.B_1,self.B_2,self.A_2])
        self.solutions0 = solve([self.A_01+self.A_02-self.B_02,
                        I*self.k_1*self.A_01-I*self.k_1*self.A_02-self.B_01,
                        self.B_01*self.a+self.B_02-self.C_01*exp(I*self.k_1*self.a),
                        self.B_01-I*self.k_1*self.C_01*exp(I*self.k_1*self.a)],
                        [self.C_01,self.B_01,self.B_02,self.A_02])
        D = simplify(Abs(self.solutions[self.C_1])**2/Abs(self.A_1)**2)
        R = simplify(Abs(self.solutions[self.A_2])**2/Abs(self.A_1)**2)
        D0 = simplify(Abs(self.solutions0[self.C_01])**2/Abs(self.A_01)**2)
        R0 = simplify(Abs(self.solutions0[self.A_02])**2/Abs(self.A_01)**2)
        print(D)
        print(R)
        print(D0)
        print(R0)
        return D,R,D0,R0
        #至此，R和D符号计算完成，接下来计算具体的值
    def CalcuNum(self,E,U,a_value,h0 = 1.05e-34,m = 9.11e-31):
        if U < E:
            k_1_value = np.sqrt(2*m*E/np.square(h0))
            k_2_value = np.sqrt(2*m*(E-U)/np.square(h0))
            D = (Abs(simplify(self.solutions[self.C_1].subs({self.k_1:k_1_value,self.k_2:k_2_value,self.a:a_value}))))**2/Abs(self.A_1)**2
            R = (Abs(simplify(self.solutions[self.A_2].subs({self.k_1:k_1_value,self.k_2:k_2_value,self.a:a_value}))))**2/Abs(self.A_1)**2
        elif U > E:
            k_1_value = np.sqrt(2*m*E/np.square(h0))
            k_2_value = np.sqrt(2*m*(U-E)/np.square(h0))
            D = (Abs(simplify(self.solutions[self.C_1].subs({self.k_1:k_1_value,self.k_2:I*k_2_value,self.a:a_value}))))**2/Abs(self.A_1)**2
            R = (Abs(simplify(self.solutions[self.A_2].subs({self.k_1:k_1_value,self.k_2:I*k_2_value,self.a:a_value}))))**2/Abs(self.A_1)**2
        else :
            k_01_value = np.sqrt(2*m*E/np.square(h0))
            D = (Abs(simplify(self.solutions0[self.C_01].subs({self.k_1:k_01_value,self.a:a_value}))))**2/Abs(self.A_01)**2
            R = (Abs(simplify(self.solutions0[self.A_02].subs({self.k_1:k_01_value,self.a:a_value}))))**2/Abs(self.A_01)**2       
        return D,R
#定义一个波设置，包含设置波的初始位置，设置波的演化
class WaveSet:
    def __init__(self,V,N,k0,Xv,dt=0.4,m=1.0e0,hbar=1.0e0,sigma=40.0):
        #依次为：对象、接收的势垒向量、网格点数量、波矢、网格点向量、时间差、质量、hbar、离散度。
        #由于计算机性能限制，有些常量必须修改，本例程力求展示波穿越势垒的变化，可以忽略一些无谓的内容
        self.cc = 0
        self.N = N
        self.X = Xv
        # self.Xv = np.linspace(-self.N//2,self.N//2,(self.N+1))#the points = 801 from -400 to 400
        self.dx = (Xv[1] - Xv[0])
        self.sigma = sigma
        self.m = m
        self.dt = dt
        self.x0 = -(round(self.N/2) - 15*sigma)#函数中心位置
        self.V = V#这个V是个向量，共有N个数据
        self.hbar = hbar
        self.k0 = k0
        self.psi_r = np.zeros((3 , N))
        self.psi_i = np.zeros((3 , N))
        self.psi_p = np.zeros( N, )
        self.I1 = range(1, N - 1)
        self.I2 = range(2, N - 0)
        self.I3 = range(0, N - 2)
    def init(self):#inition
        # print(type(self.X), self.X)
        self.psi_r = np.zeros((3 , self.N))
        self.psi_i = np.zeros((3 , self.N))
        self.psi_p = np.zeros( self.N, )
        wave = NorGaussLt(self.X,self.x0,self.sigma)
        wave2 = Plw(self.X,self.x0,self.sigma)
        #高斯波包初始化        
        self.psi_r[0,:] = np.cos(self.k0*self.X)*wave
        self.psi_r[1,:] = np.cos(self.k0*self.X)*wave
        self.psi_i[0,:] = np.sin(self.k0*self.X)*wave
        self.psi_i[1,:] = np.sin(self.k0*self.X)*wave
        #平面波包初始化
        # self.psi_r[0,:] = np.cos(self.k0*self.X)*wave2
        # self.psi_r[1,:] = np.cos(self.k0*self.X)*wave2
        # self.psi_i[0,:] = np.sin(self.k0*self.X)*wave2
        # self.psi_i[1,:] = np.sin(self.k0*self.X)*wave2

    def develop(self , num):
        A = self.hbar * self.dt / (self.m * self.dx ** 2)#e23
        B = 2 * self.dt / self.hbar
        Cv = B * self.V
        for i in range(num):
            self.psi_i[2, self.I1] = self.psi_i[0, self.I1] + A * (self.psi_r[1][self.I2] - 2 *self.psi_r[1][self.I1] + self.psi_r[1][self.I3])
            # print(self.psi_i[1])
            self.psi_i[2] -= Cv * self.psi_r[1]
            if self.cc < 10 :
                self.cc += 1
                print(self.psi_i[1])

            self.psi_r[2, self.I1] = self.psi_r[0, self.I1] - A * (self.psi_i[1][self.I2] - 2 *self.psi_i[1][self.I1] + self.psi_i[1][self.I3])
            self.psi_r[2] += Cv * self.psi_i[1]

            self.psi_r[0] = self.psi_r[1]
            self.psi_r[1] = self.psi_r[2]
            self.psi_i[0] = self.psi_i[1]
            self.psi_i[1] = self.psi_i[2]

        self.psi_p = np.sqrt(self.psi_r[1] ** 2 + self.psi_i[1] ** 2)

        return self.psi_p

class Animator:

    def __init__(self, waveset):
        self.count = 0
        self.waveset = waveset#传递对象
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(self.waveset.X.min(), self.waveset.X.max())
        #ymax = (self.waveset.psi_r[1]).max()
        self.ax.set_ylim(-1.5*((self.waveset.psi_r[1]).max()), 1.5*((self.waveset.psi_r[1]).max()))
        (self.Barrier,) = self.ax.plot([], [], c="black", alpha=1,label=r"$V(x)$",linewidth=1)
        (self.Line,) = self.ax.plot([], [], c="black", alpha=1,label=r"$V(x)$",linewidth=1,zorder=100)
        (self.realPart,) = self.ax.plot([], [], c="y",  alpha=0.5,label=r"$\psi(x)_{real}$",linewidth=1)
        (self.imaPart,) = self.ax.plot([], [], c="m",  alpha=0.5,label=r"$\psi(x)_{imaginary}$",linewidth=1)
        (self.Prob,) = self.ax.plot([], [], c="red", alpha=0.5,label=r"$|\psi(x)|$", linewidth=1)
        
        self.title = self.ax.set_title("")
        self.ax.legend(prop=dict(size=5))
        self.ax.set_xlabel("$x$")
        # self.ax.set_ylabel(r"$|\psi(x)|^2$")

        self.Barrier.set_data(self.waveset.X, self.waveset.V)
        plt.fill_between(self.waveset.X, self.waveset.V, where=self.waveset.V >= 0, facecolor='black', alpha=1)  

    def init(self):
        self.realPart.set_data([], [])
        self.imaPart.set_data([], [])
        self.Prob.set_data([], [])
        self.Line.set_data([], [])
        self.title.set_text("")
        return (self.realPart, self.imaPart, self.Prob,self.Line)

    def update(self, num):

        self.realPart.set_data(self.waveset.X, self.waveset.psi_r[2])
        self.imaPart.set_data(self.waveset.X, self.waveset.psi_i[1])
        self.Prob.set_data(self.waveset.X,  self.waveset.psi_p)
        self.Line.set_data(self.waveset.X, 0)
        self.count += 1
        # return self.realPart, self.imaPart, self.Prob

    def time_step(self):
        while True:
            if self.count == 377:
                self.waveset.init()
                self.count = 0
            yield self.waveset.develop(10)

    def animate(self, save=False,save_images=True,image_dir=None):
        self.ani = animation.FuncAnimation(self.fig, self.update, self.time_step, interval=20,save_count=299)
        # self.ani.save(filename='g.mp4',dpi=512)
        # if save:
        #     with open("index.html", "w") as f:
        #         print(self.ani.to_jshtml(), file=f)
        # if save_images:
        #     if image_dir is None:
        #         image_dir = 'animation_frames'
        #     os.makedirs(image_dir, exist_ok=True)

        # for i in range(200):  # 假设动画总帧数为200，根据实际需要调整
        #     self.ani._func(i, *self.ani._args)  # 更新图形
        #     filename = f"{i:03d}.png"  # 文件名格式为 000.png 或 000.jpg
        #     filepath = os.path.join(image_dir, filename)
        #     self.fig.savefig(filepath, dpi=300, bbox_inches='tight')  # 设置高DPI和紧致边界

        # 如果需要同时显示动画，这里可以添加plt.show()，否则仅保存图片而不显示