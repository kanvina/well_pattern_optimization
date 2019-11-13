import numpy as np
import pandas as pd


def get_array_from_csv(path):
    data_array=np.array(pd.read_csv(path,header=None))
    return data_array


def get_target_location(point_num,data_shape):
    point_row=int( np.floor(point_num/data_shape[1]))
    point_column=int(point_num%data_shape[1])
    return [point_row,point_column]


if __name__ =="__main__":

    data_H=get_array_from_csv('data/data_IDW_H(m).csv')
    data_V=get_array_from_csv('data/data_IDW_V.csv')
    data_K=get_array_from_csv('data/data_IDW_K.csv')
    data_p=get_array_from_csv('data/data_IDW_p(Mpa).csv')

    data_shape=np.shape(data_H)

    data_target=np.zeros(data_shape)

    for point_num in range(data_shape[0]*data_shape[1]):
        [point_row,point_column]=get_target_location(point_num, data_shape)

        Value_B=10**3
        Value_H=data_H[point_row,point_column]
        Value_V=data_V[point_row,point_column]
        Value_K=data_K[point_row,point_column]
        Value_p=data_p[point_row,point_column]
        Value_p0=1.5


        Value_target=Value_B*Value_H*Value_V*Value_K*0.01*(Value_p**2-Value_p0**2)

        data_target[point_row,point_column]=Value_target

    pd.DataFrame(data_target).to_csv('data/data_target.csv',index=0,header=0)





