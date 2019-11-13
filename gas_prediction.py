import numpy as np
from sympy import *
'''
物质平衡法预测产能
'''

class Gas_prediction():

    def __init__(self):
        self.P_L=2.12#Langmuir 压力系数，Mpa
        self.V_L=10.79#Langmuir 体积系数，m^3/t
        self.P_i=6.9 #初始压力，Mpa
        self.A=90000 #供气面积，m^2
        self.h=6 #煤厚
        self.phi_i=0.01#初始孔隙度
        self.T_sc=273.15#标准温度，K
        self.P_sc=0.101#标准压力，Mpa
        self.Z_sc=1#标准气体压缩因子
        self.B_g=1#气体地层体积系数
        self.Z=1#气体压缩因子
        self.c_f=1#底层压缩系数
        self.rho_B=1.58#煤密度，t/m^3
        self.S_wi=0.95#初始含水饱和度，%
        self.c_W=4.7*10**(-4)#水压缩系数，1/MPa
        self.B_W=1#水的地层体积系数
        self.T=293#温度，K
        self.W_e=0#侵入的水量，m3


    def get_z_star(self):
        pass

    def get_P(self,W_p,G_p):
        P = Symbol('P')
        Z_star_i=self.Z/( (self.rho_B*self.B_g*self.V_L*self.P_i)/(self.phi_i*(self.P_L+self.P_i)) )
        A=(self.P_i/Z_star_i)-( G_p*self.P_sc*self.T/(self.A*self.h*self.phi_i*self.T_sc*self.Z_sc) )
        S_w=(self.S_wi*(1+self.c_W*(self.P_i-P))+(5.615*(self.W_e-(self.B_W*W_p))/(self.phi_i*self.A*self.h)))/(1-self.c_f*(self.P_i-P))
        Z_star=self.Z/( (1-self.c_f*(self.P_i-P))*(1-S_w)+( self.rho_B*self.B_g*self.V_L*P )/( self.phi_i*(self.P_L+P)))
        equation=Z_star*A-P
        result_P = solve(equation, P)

        return result_P




if __name__ =="__main__":
    GP=Gas_prediction()
    result_P=GP.get_P(0,0)

    print(result_P)