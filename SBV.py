import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from sympy import symbols , lambdify , sympify
from D3 import *

if __name__=="__main__":
    N = 601
    x = symbols('x')
    # func = -3e-3*(x+0)*(x-10)
    # func = 1e-2*x
    func = 3e-2
    func = lambdify(x,sympify(func))
    width = 10
    xlim = 300
    V = GeBarrier(width = width , func = func , Xv=np.linspace(-xlim,xlim,N))
    """
    Normal so far.You could plt the picture of V latter.
    """
    waveset = WaveSet(V=V,N=N,k0=(np.pi)/20,Xv=np.linspace(-xlim,xlim,N),m=1,hbar=1,sigma=10,)
    waveset.init()
    animator = Animator(waveset)
    animator.animate(save = False,save_images=False)
    # writer = FFMpegWriter(fps=24,bitrate=8000)  # 设置帧率
    # animator.ani.save('animation.mp4', writer=writer,dpi=1024)
    plt.show()


