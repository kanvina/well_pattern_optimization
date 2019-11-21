import numpy as np
from sympy import *
from matplotlib import pyplot as plt
'''
物质平衡法预测产能
'''

class Gas_prediction():

    def __init__(self):
        self.P_L=3.5       #Langmuir 压力系数，Mpa
        self.V_L=27    #Langmuir 体积系数，m^3/t
        self.P_i=5    #初始压力，Mpa
        self.A=90000       #供气面积，m^2
        self.h=6           #煤厚
        self.phi_i=0.045   #初始孔隙度
        self.T_sc=273.15   #标准温度，K
        self.P_sc=14.75   #标准压力，Mpa
        self.Z_sc=1        #标准气体压缩因子
        self.B_g=1         #气体地层体积系数
        self.Z=0.9       #气体压缩因子
        self.c_f=0.0005      #地层压缩系数
        self.c_d=0.0005
        self.rho_B=1.58    #煤密度，t/m^3
        self.S_wi=0.95     #初始含水饱和度
        self.c_W=0.0004    #水压缩系数，1/MPa
        self.B_W=1         #水的地层体积系数
        self.T=318         #温度，K
        self.W_e=0         #侵入的水量，m3
        self.P_b=0         #任意基座压力，Mpa
        self.P_wf=0.5     #井底流压
        self.q_g=2.5       #气体相对渗透率
        self.mu_g=0.01    #气体粘度，mPa/s
        self.r_e=300     #泄流半径，m
        self.r_w=1         #井筒半径，m
        self.s=-3           #表皮系数
        self.k_g=0.63      #气体有效渗透率
        self.k_w=2.5       #水的有效渗透率
        self.mu_w=0.678      #液体粘度
        self.m_P_wf=self.m(self.P_wf)
        self.G_c=20#含气量
        self.W_t=0


    def get_P(self,W_p,G_p):
        P = Symbol('P')
        Z_star_i=self.Z/( (self.rho_B*self.B_g*self.V_L*self.P_i)/(self.phi_i*(self.P_L+self.P_i)) )
        A=(self.P_i/Z_star_i)-( G_p*self.P_sc*self.T/(self.A*self.h*self.phi_i*self.T_sc*self.Z_sc) )
        S_w = self.S_wi * self.B_W * W_p / (7758.4 * self.A * self.h * self.phi_i)
        Z_star=self.Z/( (1-self.c_f*(self.P_i-P))*(1-S_w)+( self.rho_B*self.B_g*self.V_L*P )/( self.phi_i*(self.P_L+P)))
        equation=Z_star*A-P
        result_P = solve(equation, P)

        result_P=result_P[-1]
        result_P_str=str(result_P).replace('-','+')
        result_P_float=float(result_P_str.split('+')[0])
        return result_P_float

    def get_P_v2(self, W_p, G_p):
        P = Symbol('P')
        S_w = (self.S_wi * (1 + self.c_W * (self.P_i - P)) + (5.615 * (self.W_e - (self.B_W * W_p)) / (self.phi_i * self.A * self.h))) / (1 - self.c_f * (self.P_i - P))
        V_p=self.A*self.h*self.rho_B
        G = V_p * self.G_c
        Dealt_V_p=( self.W_e-self.W_t-W_p*self.B_W+V_p*self.c_W*(1-S_w)*(self.P_i-P) )+V_p*(self.S_wi-S_w)+( V_p*self.c_d*(self.P_i-P)/self.phi_i )-V_p*self.c_f*(self.P_i-P)
        equation=G+V_p*(1-self.S_wi)*( self.T_sc/(self.P_sc*P))*(self.P_i/self.Z)-G*(self.V_L/self.G_c)*(P/(self.P_L+P))-(1-S_w)*(V_p)*( self.T_sc/(self.P_sc*self.T) )*(P/self.Z)-G_p
        result_P = solve(equation, P)

        print(result_P)
        P_result=0

        for P_line in result_P:
            P_line=P_line.evalf()
            if P_line>0 and P_line<=20:
                P_result=P_line



        return P_result



    # def get_P_without_water(self,G_p):
    #     P=self.P_L*(self.G_c*self.A*self.h*self.rho_B-G_p  )/( self.A*self.h*self.rho_B*self.V_L-self.G_c*self.A*self.h*self.rho_B+G_p  )
    #     return  P
    def get_P_without_water(self,G_p):
        G_i = self.G_c * self.A * self.h * self.rho_B
        P=self.P_L*self.P_i*( 1-(G_p/G_i) )/self.P_i*( 1-(G_p/G_i) )
        return  P


    def m(self,P_in):
        m_p=2*(P_in**2- self.P_b**2)/( 2*self.mu_g* self.Z)
        return m_p


    def get_gas_prediction(self,P,k_g):
        m_p=self.m(P)
        q_g=1000*( k_g*self.h*(m_p-self.m_P_wf))/( 1.31*self.T*(  np.log(self.r_e/self.r_w)-0.75+self.s  ) )

        return q_g

    def get_water_prediction(self,P,k_w):
        q_w=k_w*self.h*(P-self.P_wf)/( 141.2*self.mu_w*self.B_W*(np.log(self.r_e/self.r_w)-0.75+self.s) )
        q_w=100000

        return q_w


    def get_k_rg_k_rw(self,S_w):
        k_rg=(1-S_w)**2
        k_rw=S_w**2
        return k_rg,k_rw

    def get_S_w(self,W_p):
        S_w = self.S_wi - (self.B_W * W_p / (7758.4 * self.A * self.h * self.phi_i))
        return S_w



if __name__ =="__main__":
    GP = Gas_prediction()
    W_p=0
    G_p=0


    q_g_list=[]
    q_w_list=[]
    i_list=[]
    for i in range(540):
        # s_w=GP.get_S_w(W_p)
        # print(s_w)
        print(i)
        i_list.append(i)
        result_P=GP.get_P_without_water(G_p)
        print('压力： ',result_P)

        S_w=GP.get_S_w(W_p)
        k_rg, k_rw=GP.get_k_rg_k_rw(S_w)

        q_g=GP.get_gas_prediction(result_P,k_rg)
        q_g_list.append(q_g)
        print('日产气量： ',q_g)
        q_w=GP.get_water_prediction(result_P,k_rw)
        print('日产水量： ',q_w)
        q_w_list.append(q_w)

        W_p=W_p+q_w*10
        G_p=G_p+q_g*10
    print(G_p)
    plt.scatter(i_list, q_g_list, marker='o', color='red', s=40, label='First')
    # plt.scatter(i_list, q_w_list, marker='x', color='blue', s=40, label='First')
    plt.show()




