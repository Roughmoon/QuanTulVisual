# 量子隧道效应

​	量子隧穿(部分教材或文献称之为量子贯穿或量子隧道等)是一种典型的量子效应，简单来说，就是当入射粒子能量小于势垒高度时，入射粒子仍然有一定的概率穿过势垒。当入射粒子能量高于势垒高度，仍然有一定概率会在势垒表面反射。这些现象是经典力学无法解释的，需要借助于量子力学理论，本质为微观粒子具有波粒二象性。

# 势垒贯穿物理原理

## 初始波函数设定及其约束

### 初始波函数设定

$\psi\left( {x,0} \right) = \frac{1}{\left( 2\pi\sigma_{x}^{2})^{\frac{1}{4}} \right.}e^{- (x - x_{0})^{2}/(2\sigma_{x})^{2}}e^{i\frac{p_{0}}{\hslash}x}$

这是在$t=0$时刻的波函数，其中的参数包括$x_0,\sigma_x,p_0$

### 约束情况

![1731167016586.png](https://img.picui.cn/free/2024/11/09/672f8329a7d2b.png)

根据以上图片及势垒约束，分成三个区域，这三个区域中包含着不同的三个波函数（分别为$\psi_1,\psi_2,\psi_3$，薛定谔说，这三个区域的波函数都满足他的方程，结果如下：



$$
\begin{array}{l}
-\frac{h^{2}}{2 m} \frac{\mathrm{~d}^{2} \psi_{1}}{\mathrm{~d} x^{2}}=E \psi_{1}(x<0) \\
-\frac{h^{2}}{2 m} \frac{\mathrm{~d}^{2} \psi_{2}}{\mathrm{~d} x^{2}}+U_{0} \psi_{2}=E \psi_{2}(0<x<a) \\
-\frac{h^{2}}{2 m} \frac{\mathrm{~d}^{2} \psi_{3}}{\mathrm{~d} x^{2}}=E \psi_{3}(x>a)
\end{array}
$￥



有时，为了方便运算，我们记几个符号：
$k_1=\frac{\sqrt{2mE}}{h},k_2=\frac{\sqrt{2m(E-V_0)}}{h},k_3=\frac{\sqrt{2m(V_0-E)}}{h},ik_3=k_2$

这时候，薛定谔给的方程就可以携写成：

$$
\begin{array}{l}
\frac{d^{2} \psi_{1}}{d x^{2}}+k_{1}^{2} \psi_{1}=0 \\
\frac{d^{2} \psi_{2}}{d x^{2}}+k_{2}^{2} \psi_{2}=0 \\
\frac{d^{2} \psi_{3}}{d x^{2}}+k_{1}^{2} \psi_{3}=0
\end{array}
$$

# 文件说明

视频文件为最终效果文件.mp4

SBV.py为实现视频效果的python文件

SBDR.py为求解波函数的符号解的python文件

其余文件为辅助文件或过渡文件



