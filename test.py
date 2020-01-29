from gas_prediction import run
import copy
import numpy as np
import pandas as pd
if __name__=="__main__":
    # well_info = {
    #     'A': 40000 * 3.1415926,
    #     'P_L': 4,
    #     'P_cd': 3.5,
    #     'V_L': 24.75,
    #     'P_i': 6,
    #     'h': 15,
    #     'phi_i': 0.01,
    #     'K_i': 2,
    #     'rho_B': 1.51
    # }
    #
    # B=len(well_info)
    # name=list(well_info.keys())[0]


    a=[1,2,3]
    b=[4,6,5]
    c=list(zip(a,b))
    pd.DataFrame(c).to_csv('data/test.csv')

    print(1)

