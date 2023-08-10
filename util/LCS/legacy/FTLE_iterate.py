#-*-coding:utf-8-*-
'''
多变量非线性方程求解
'''
from sympy import *
import numpy as np
# np.set_printoptions(suppress=True)
import copy

n = 5000#控制迭代次数


def Henon(x, y, n):
    for i in range(n):
        x1 = 1 - 1.4 * x ** 2 + y
        y1 = 0.3 * x
        x = x1
        y = y1
    return x, y


def LE_calculate():
    sum_Lambda1 = 0
    sum_Lambda2 = 0
    a = 0.123456789
    b = 0.123456789
    # 使用符号方式求解
    x, y = symbols("x,y")
    f_mat = Matrix([1 - 1.4 * x ** 2 + y, 0.3 * x])
    # 求解雅各比矩阵
    jacobi_mat = f_mat.jacobian([x, y])  # 带变量的雅各比矩阵形式是固定的
    a, b = Henon(a, b, 5001)  # 先迭代5000次，消除初始影响.以第5001次的值作为初始值
    U1 = Matrix([1, 0])  # 初始列向量
    U2 = Matrix([0, 1])
    for i in range(n):
        J = jacobi_mat.subs({x: a, y: b})  # 将变量替换为当前迭代值，得到当前的雅各比矩阵（数字）
        column_vector1 = U1#初始列向量为上一次的U1和U2
        column_vector2 = U2
        vector1 = J * column_vector1  # 初始列向量乘上雅各比矩阵之后得到的向量
        vector2 = J * column_vector2
        V1 = vector1  # 将vector1复制给V1
        U1 = V1 / (V1.norm(2))  # 向量U1等于向量V1除以向量V1的模(2范数)
        V2 = vector2 - (vector2.dot(U1)) * U1  # dot为点乘(內积)
        U2 = V2 / (V2.norm(2))
        Lambda1 = ln(V1.norm(2))
        Lambda2 = ln(V2.norm(2))
        sum_Lambda1 = sum_Lambda1 + Lambda1
        sum_Lambda2 = sum_Lambda2 + Lambda2
        a, b = Henon(a,b,1)#进行下一次迭代

    LE1=sum_Lambda1/n
    LE2=sum_Lambda2/n
    print(LE1)
    print(LE2)

if __name__ == '__main__':
    LE_calculate()
