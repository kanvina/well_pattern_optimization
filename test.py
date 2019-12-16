
import numpy as np
from gas_prediction_V2 import  Gas_prediction
from matplotlib import pyplot as plt
def get_A(r,l):
    theta=np.arccos(l/(2*r))* 180 / np.pi
    A=11*(3.1415926*r**2*2-2*(3.1415926*r**2*2*theta/360-(l*r*np.sin(theta*np.pi/180))/2))
    return A




q_g_list = []
q_w_list = []
P_list = []
i_list = []
Z_list = []
phi_list = []
S_w_list = []
G_P_list = []
P_wf_list = []


for l in range(0,410,50):
    i_list.append(l)
    a = get_A(200,l)

    GP = Gas_prediction(a)

    P = GP.P_i
    W_p = 0
    G_p = 0
    Z = GP.get_z(P * 0.006895, GP.T / 1.8, 0.8)
    phi = GP.phi_i
    S_w = GP.S_wi
    k_g, k_w = GP.get_k_rg_k_rw(S_w)
    P_wf = GP.get_P_wf(P, k_w)
    G_p_L_1 = 0
    for i in range(7200):

        if P_wf > GP.P_wf:
            q_w = GP.q_wi * 2
        else:
            q_w = GP.get_water_prediction(P, k_w, P_wf) * 2

        q_w_list.append(q_w * 0.159)

        W_p = W_p + q_w

        Z = GP.get_z(P * 0.006895, GP.T / 1.8, 0.8)
        Z_list.append(Z)


        phi = GP.get_phi(P)
        phi_list.append(phi)


        S_w = GP.get_S_w(W_p, phi)
        S_w_list.append(S_w)


        k_g, k_w = GP.get_k_rg_k_rw(S_w)

        P_wf = GP.get_P_wf(P, k_w)
        if P_wf <= GP.P_wf:
            P_wf = GP.P_wf
        P_wf_list.append(P_wf / 145)

        if P > GP.P_cd:
            P = GP.get_P_1(S_w, Z, phi, G_p)
        else:
            P = GP.get_P_2(S_w, Z, phi, G_p)

        P_list.append(P / 145)


        if P > GP.P_cd:
            q_g = GP.get_gas_prediction_level_1(P, phi, S_w, Z, G_p) * 2
        else:
            q_g = GP.get_gas_prediction(P, k_g, Z, P_wf) * 2

        q_g_list.append(q_g * 10 ** 6 * 0.028)

        G_p = G_p + q_g
    print('l:',l,'G_p', G_p * 10 ** 6 * 0.028)
    G_P_list.append(G_p * 10 ** 6 * 0.028)

plt.scatter(i_list, G_P_list, marker='x', color='red', s=10, label='First')
plt.show()

