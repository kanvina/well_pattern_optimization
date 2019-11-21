import numpy as np
# from sympy import *
from matplotlib import pyplot as plt
'''
物质平衡法预测产能
'''

class Gas_prediction():

    def __init__(self):
        self.P_L=759   #Langmuir 压力系数，Mpa
        self.V_L=920 #Langmuir 体积系数，m^3/t
        self.P_i=1191   #初始压力，Mpa
        self.A=40000*11    #供气面积，m^2
        self.h=6*3.3        #煤厚
        self.phi_i=0.012  #初始孔隙度
        self.Z=0.9       #煤层气偏差系数
        self.rho_B=1.58 /50 #煤密度，t/m^3
        self.S_wi=0.95     #初始含水饱和度
        self.B_W=1         #水的地层体积系数
        self.T=70        #温度，K
        self.P_b=0         #任意基座压力，Mpa
        self.P_wf=100     #井底流压
        self.mu_g=0.01    #气体粘度，mPa/s
        self.r_e=330     #泄流半径，m
        self.r_w=0.2       #井筒半径，m
        self.s=-3           #表皮系数
        self.G_c=20*37        #含气量


    def get_P_without_water(self,G_p):
        G_i = self.G_c * self.A * self.h * self.rho_B
        P=self.P_L*self.P_i*( 1-(G_p/G_i) )/self.P_i*( 1-(G_p/G_i) )
        return  P
    # def get_P(self, G_p,S_w):
    #     P = Symbol('P')
    #     Z_star_i = self.Z / ((self.rho_B * self.B_g * self.V_L * self.P_i) / (self.phi_i * (self.P_L + self.P_i)))
    #     Z_star = self.Z / ((1 - self.c_f * (self.P_i - P)) * (1 - S_w) + (self.rho_B * self.B_g * self.V_L * P) / (self.phi_i * (self.P_L + P)))
    #     G_i=self.G_c*self.A*self.h*self.rho_B
    #
    #     equation =Z_star*self.P_i*( 1-(G_p/G_i) )/Z_star_i-P
    #     result_P = solve(equation, P)
    #
    #     result_P = result_P[-1]
    #     result_P_str = str(result_P).replace('-', '+')
    #     result_P_float = float(result_P_str.split('+')[0])
    #     return result_P_float

    def m(self,P_in):
        m_p=(P_in**2- self.P_b**2)/( self.mu_g* self.Z)
        return m_p


    def get_gas_prediction(self,P,k_g):
        m_p=self.m(P)
        m_P_wf=self.m(self.P_wf)

        q_g=1000*( k_g*self.h*(m_p-m_P_wf))/( 1422*self.T*(  np.log(self.r_e/self.r_w)-0.75+self.s  ) )
        return q_g

    def get_water_prediction(self,P,k_w):
        q_w=k_w*self.h*(P-self.P_wf)/( 141.2*self.mu_w*self.B_W*(np.log(self.r_e/self.r_w)-0.75+self.s) )
        q_w=10000

        return q_w


    def get_k_rg_k_rw(self,S_w):
        k_rg=(1-S_w)**1.8
        k_rw=S_w**1.8
        return k_rg,k_rw

    def get_S_w(self,W_p):
        S_w = self.S_wi -( self.B_W * W_p / (7758.4 * self.A * self.h * self.phi_i))
        return S_w







if __name__ =="__main__":
    GP = Gas_prediction()
    W_p=0
    G_p=0
    q_w=1000*37


    q_g_list=[]
    q_w_list=[]
    P_list=[]
    i_list=[]

    for i in range(5400):
        print(i)
        i_list.append(i)

        P=GP.get_P_without_water(G_p)
        P_list.append(P)
        print('P:',P)

        S_w = GP.get_S_w(W_p)
        k_g, k_w=GP.get_k_rg_k_rw(S_w)

        q_g=GP.get_gas_prediction(P,k_g)
        q_g_list.append(q_g)
        print('q_g:',q_g)

        G_p=G_p+q_g
        W_p=W_p+q_w
    plt.scatter(i_list, q_g_list, marker='o', color='red', s=40, label='First')
    # plt.scatter(i_list, P_list, marker='x', color='blue', s=40, label='First')
    plt.show()



