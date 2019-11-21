from sympy import *


def m(P_in):
    P = symbols('P')
    P_b=0
    m_p = 2 * float(integrate(P / (0.002  *0.12  ), (P, P_b, P_in)))
    return m_p

