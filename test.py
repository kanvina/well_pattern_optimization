import numpy as np
# from sympy import *
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

'''
物质平衡法预测产能
'''


class Gas_prediction():

    def __init__(self):
        self.P_L = 2.38 * 145  # Langmuir 压力系数，Mpa
        self.P_d = 3.5 * 145  # 吸附压力
        self.V_L = 38.16 * 37  # Langmuir 体积系数，m^3/t
        self.P_i = 5 * 145  # 初始压力，Mpa
        self.A = 40000 * 11  # 供气面积，m^2
        self.h = 15 * 3.3  # 煤厚
        self.phi_i = 0.01  # 初始孔隙度
        self.rho_B = 1.45 * 60  # 煤密度，t/m^3
        self.S_wi = 0.95  # 初始含水饱和度
        self.B_W = 1  # 水的地层体积系数
        self.T = 313 * 1.8  # 温度，K
        self.P_b = 0  # 任意基座压力，Mpa
        self.P_wf = 1.1 * 145  # 井底流压
        self.mu_g = 0.01  # 气体粘度，mPa/s
        self.r_e = 300 * 3.3  # 泄流半径，m
        self.r_w = 0.1 * 3.3  # 井筒半径，m
        self.s = -3  # 表皮系数
        self.G_c = 14.1 * 0.0168  # 含气量
        self.mu_w = 0.6
        self.P_sc = 14.7
        self.T_sc = 520
        self.Z_i = self.get_z(self.P_i * 0.006895, self.T / 1.8, 0.8)
        self.B_gi = self.Z_i * self.T * self.P_sc / (self.P_i * self.T_sc)
        self.G_f = self.A * self.h * self.phi_i * (1 - self.S_wi) / self.B_gi

    def get_z(self, P, T, theta):
        '''
        算法来源：天然气压缩因子计算方法对比及应用_董萌
        :param P:
        :param theta: 天然气相对密度
        :return:
        '''
        P = P * 1000

        A1 = 0.31506237
        A2 = -1.046099
        A3 = -0.57832720
        A4 = 0.53530771
        A5 = -0.61232023
        A6 = -0.10488813
        A7 = 0.68127001
        A8 = 0.68446549
        P_c = 4881 - 386.11 * theta
        T_c = 92 + 116.67 * theta
        P_r = P / P_c
        T_r = T / T_c

        T1 = A1 + (A2 / T_r) + (A3 / T_r ** 3)
        T2 = A4 + (A5 / T_r)
        T3 = A5 * A6 / T_r
        T4 = A7 / (T_r ** 3)
        T5 = 0.27 * P_r / T_r

        rho = 0.27 * P_r / T_r
        rho_pass = False

        while rho_pass == False:
            f_rho = 1 + T1 * rho + T2 * rho ** 2 + T3 * rho ** 5 + (
                        T4 * rho ** 2 * (1 + A8 * rho ** 2) * np.exp(-(A8 * rho ** 2))) - T5 / rho
            f_rho_coe = T1 + 2 * T2 * rho + 5 * T3 * rho ** 4 + 2 * T4 * rho * (
                        1 + A8 * rho ** 2 - A8 ** 2 * rho ** 4) * np.exp(-(A8 * rho ** 2)) + T5 / rho ** 2

            rho_old = rho
            rho_new = rho - (f_rho / f_rho_coe)
            rho = rho_new
            if rho_old - rho_new < 0.1:
                rho_pass = True
        Z = 0.27 * P_r / (rho_new * T_r)
        return Z

    def get_P_without_water(self, G_p):
        G_i = self.G_c * self.A * self.h * self.rho_B / 1000000
        A = 1 - (G_p / G_i)
        P = self.P_L * self.P_i * (1 - (G_p / G_i)) / (self.P_i + self.P_L - self.P_i * (1 - (G_p / G_i)))
        return P

    def get_P_level_1(self, S_w, Z, phi, G_p):
        P = (self.G_f - G_p) * Z * self.T * self.P_sc / (self.A * self.h * phi * (1 - S_w) * self.T_sc)
        return P

    def m(self, P_in, Z):
        m_p = (P_in ** 2 - self.P_b ** 2) / (self.mu_g * Z)
        return m_p

    def get_gas_prediction(self, P, k_g, Z):
        m_p = self.m(P, Z)
        m_P_wf = self.m(self.P_wf, Z)

        q_g = 0.001 * (k_g * self.h * (m_p - m_P_wf)) / (1422 * self.T * (np.log(self.r_e / self.r_w) - 0.75 + self.s))
        return q_g

    def get_water_prediction(self, P, k_w):
        q_w = k_w * self.h * (P - self.P_wf) / (
                    141.2 * self.mu_w * self.B_W * (np.log(self.r_e / self.r_w) - 0.75 + self.s))
        return q_w

    def get_k_rg_k_rw(self, S_w):
        k_rg = (1 - S_w) ** 2
        k_rw = S_w ** 2
        return k_rg, k_rw

    def get_S_w(self, W_p, phi):
        S_w = self.S_wi - (self.B_W * W_p / (7758.4 * (self.A * 0.00002295684) * self.h * phi))
        return S_w

    def get_phi(self, P):
        '''
        来源：
        1:Palmer I，Mansoori J． How Permeability Depends on Stress and Pore Pressure in Coalbeds: a New Model［J］
        2:Coalbed Methane Production System Simulation and Deliverability Forecasting: Coupled Surface Network/Wellbore/Reservoir Calculation
        :param P:
        :return:
        '''
        phi = self.phi_i + 3 * 10 ** (-6) * (P - self.P_i) - 0.0064 * (
                    (P / (self.P_L + P)) - (self.P_i / (self.P_L + self.P_i)))
        return phi


if __name__ == "__main__":
    GP = Gas_prediction()
    W_p = 0
    G_p = 0

    q_g_list = []
    q_w_list = []
    P_list = []
    i_list = []
    Z_list = []
    phi_list = []
    S_w_list = []

    P = GP.P_i

    for i in range(7200):
        print(i + 1)
        i_list.append(i)

        P = GP.get_P_without_water(G_p)
        P_list.append(P / 145)
        print('P:', P / 145)

        Z = GP.get_z(P * 0.006895, GP.T / 1.8, 0.8)
        Z_list.append(Z)
        print('Z:', Z)

        phi = GP.get_phi(P)
        phi_list.append(phi)
        print('phi:', phi)

        S_w = GP.get_S_w(W_p, phi)
        S_w_list.append(S_w)
        print('S_w:', S_w)

        k_g, k_w = GP.get_k_rg_k_rw(S_w)

        q_g = GP.get_gas_prediction(P, k_g, Z)
        q_g_list.append(q_g * 10 ** 6 * 0.028)
        print('q_g:', q_g * 10 ** 6 * 0.028)

        if P > GP.P_d:
            q_w = 2.1 * 6.289
        else:
            q_w = GP.get_water_prediction(P, k_w)

        # q_w = GP.get_water_prediction(P, k_w)

        q_w_list.append(q_w * 0.159)
        print('q_w:', q_w * 0.159)

        G_p = G_p + q_g
        W_p = W_p + q_w

    fig = plt.figure()
    font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc")

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title('日产水量', fontproperties=font)
    plt.scatter(i_list, q_w_list, marker='x', color='red', s=10, label='First')

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title('日产气量', fontproperties=font)
    plt.scatter(i_list, q_g_list, marker='o', color='red', s=10, label='First')

    ax3 = fig.add_subplot(3, 2, 3)
    ax3.set_title('压力', fontproperties=font)
    plt.scatter(i_list, P_list, marker='x', color='blue', s=10, label='First')

    ax4 = fig.add_subplot(3, 2, 4)
    ax4.set_title('Z', fontproperties=font)
    plt.scatter(i_list, Z_list, marker='x', color='red', s=10, label='First')

    ax5 = fig.add_subplot(3, 2, 5)
    ax5.set_title('孔隙度', fontproperties=font)
    plt.scatter(i_list, phi_list, marker='x', color='red', s=10, label='First')

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.set_title('含水饱和度', fontproperties=font)
    plt.scatter(i_list, S_w_list, marker='x', color='red', s=10, label='First')

    plt.show()
