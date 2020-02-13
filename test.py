'''
create by WYD
2019 10 25
井网构造实验
1- 构造原始井网单元
2- 横向与纵向井距，因子：x,y
3- 横向与纵向平移，因子：Delta_x，
4- 井网单元形状改变，因子：夹角theta，Delta_y
5- 旋转，因子：gamma

'''

import numpy as np
import pandas as pd
import copy
from gas_prediction import Gas_prediction

from matplotlib import pyplot as plt

class well_grid_class():

    def __init__(self,grid_info):

        self.x = grid_info['x']      #横轴缩放因子
        self.y= grid_info['y']        #纵轴缩放因子
        self.theta = grid_info['theta']        #井组夹角
        self.Delta_x = grid_info['Delta_x']      #横轴移动因子
        self.Delta_y = grid_info['Delta_y']      #纵轴移动因子
        self.gamma=grid_info['gamma']

    def grid_rotate(self, grid_center_point_location, point):
        '''
        井网旋转函数
        :param grid_center_point_location: 旋转中心点
        :param point: 待旋转点
        :return:
        '''
        # 待旋转点与中心点距离
        distance = ((point[1] - grid_center_point_location[1]) ** 2 + (
                point[0] - grid_center_point_location[0]) ** 2) ** 0.5
        if distance == 0:
            distance = 1
        try:
            # 第三象限
            if point[1] - grid_center_point_location[1] <= 0 and point[0] - grid_center_point_location[0] <= 0:
                angle = np.arcsin(np.abs(point[1] - grid_center_point_location[1]) / distance) * 180 / np.pi + 180
            # 第一象限
            elif point[1] - grid_center_point_location[1] >= 0 and point[0] - grid_center_point_location[0] >= 0:
                angle = np.arcsin(np.abs(point[1] - grid_center_point_location[1]) / distance) * 180 / np.pi
            # 第二象限
            elif point[1] - grid_center_point_location[1] >= 0 and point[0] - grid_center_point_location[0] <= 0:
                angle = 180 - np.arcsin(np.abs(point[1] - grid_center_point_location[1]) / distance) * 180 / np.pi
            # 第四象限
            elif point[1] - grid_center_point_location[1] <= 0 and point[0] - grid_center_point_location[0] >= 0:
                angle = -np.arcsin(np.abs(point[1] - grid_center_point_location[1]) / distance) * 180 / np.pi
        except:
            print(point, "点旋转出现异常")
        point_rotate_location = [distance * np.cos((angle - self.gamma) * np.pi / 180) + grid_center_point_location[0],
                                 distance * np.sin((angle - self.gamma) * np.pi / 180) + grid_center_point_location[1]]

        return point_rotate_location

    def create_rhombus_well_grid(self,range_x,range_y):
        grid_len = 1.4 * max(range_x[1] - range_x[0], range_y[1] - range_y[0])+2*max(self.x,self.y)

        center_location = [ self.Delta_x+(range_x[1] + range_x[0]) / 2, self.Delta_y+(range_y[1] + range_y[0]) / 2]
        range_x = [center_location[0] - grid_len / 2, center_location[0] + grid_len / 2]
        range_y = [center_location[1] - grid_len / 2, center_location[1] + grid_len / 2]

        x_num=0
        y_num=0
        x_len=self.x
        y_len=self.y
        points_array={}

        while y_num * y_len * np.sin(self.theta * np.pi / 180) + range_y[0] <= range_y[1]:
            points_array[y_num] = []

            while x_len * x_num + range_x[0] <= (range_x[1] + x_len):

                if y_num == 0:

                    point_LT_location = [y_len * np.cos(self.theta * np.pi / 180) + x_num * x_len + range_x[0],
                                         y_len * np.sin(self.theta * np.pi / 180) + range_y[0]]
                    points_array[y_num].append([y_num, x_num, point_LT_location])
                    x_num = x_num + 1

                else:

                    later_start_x = points_array[y_num - 1][0][2][0]
                    start_x_location = later_start_x - (np.floor((later_start_x - range_x[0]) / x_len) + 1) * x_len
                    point_LT_location = [start_x_location + y_len * np.cos(self.theta * np.pi / 180) + x_num * x_len,
                                         y_len * np.sin(self.theta * np.pi / 180) + y_num * y_len * np.sin(
                                             self.theta * np.pi / 180) + range_y[0]]
                    points_array[y_num].append([y_num, x_num, point_LT_location])
                    x_num = x_num + 1
            else:
                y_num = y_num + 1
                x_num = 0
        else:
            pass

        grid_center_point_location = [(range_x[1] + range_x[0]) / 2, (range_y[1] + range_y[0]) / 2]

        for points_row in points_array:
            points_line = points_array[points_row]
            for point in points_line:
                point_LT = point[2]
                point_LT_rotate = self.grid_rotate(grid_center_point_location, point_LT)
                point[2] = [point_LT_rotate[0] , point_LT_rotate[1]]
        return points_array



# def draw_scatter(points_array,xlim,ylim):
#     for point_location in points_array:
#         plt.scatter(point_location[0], point_location[1], marker='o', color='red', s=5, label='First')
#     plt.plot([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]],[ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]])
#         # plt.xlim(xlim)
#         # plt.ylim(ylim)
#     plt.show()

def draw_scatter(points_array,xlim,ylim):
    for point_location in points_array:
        plt.scatter((point_location[0]-xlim[0])/100, (point_location[1]-ylim[0])/100, marker='o', color='red', s=5, label='First')
    # plt.plot([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]],[ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]])
    plt.xlim([0,(xlim[1]-xlim[0])/100])
    plt.ylim([0,(ylim[1]-ylim[0])/100])
    plt.axis('off')
    plt.show()

def create_grid(grid_info,range_x,range_y,compute_range_x,compute_range_y):

    grid=well_grid_class(grid_info)
    points_array=grid.create_rhombus_well_grid(range_x,range_y)

    well_points_array=[]

    for points_row in points_array:
        points_line = points_array[points_row]
        for point in points_line:
            point_LT = point[2]
            if point_LT[0]>compute_range_x[0] and point_LT[0]<compute_range_x[1] and point_LT[1] >compute_range_y[0] and point_LT[1] <compute_range_y[1]:
                well_points_array.append(point_LT)

    return well_points_array

def get_cell_dict(data_array,x_range,y_range,cell_len,location):
    row=int( np.floor((y_range[1]-location[1])/cell_len))
    column=int(np.floor( (location[0]-x_range[0])/cell_len ))
    point_value=data_array[row,column]
    info_dict = eval(point_value)
    return info_dict

def get_cell_value(data_array,x_range,y_range,cell_len,location):
    row=int( np.floor((y_range[1]-location[1])/cell_len))
    column=int(np.floor( (location[0]-x_range[0])/cell_len ))
    point_value=data_array[row,column]
    return point_value

def well_gas_prediction(well_info,time):
    GP = Gas_prediction(well_info)

    '''
    定义列表，存放结果
    '''
    q_g_list=[]
    q_w_list=[]
    P_list=[]
    i_list=[]
    Z_list=[]
    phi_list=[]
    S_w_list=[]
    G_P_list=[]
    P_wf_list=[]
    K_list=[]
    Kg_list=[]

    '''
    定义初始参数
    '''
    P=GP.P_i
    W_p=0
    G_p=0
    Z = GP.Z_i
    phi = GP.phi_i
    K=GP.K_i
    S_w =GP.S_wi
    k_g, k_w = GP.get_k_rg_k_rw(S_w)
    P_wf = GP.get_P_wf( P,k_w,K)
    '''
    设定排采时间，动态预测
    '''
    for i in range(time):
        # print(i+1)
        i_list.append(i*30)
        '''
        排水，根据井底流压计算排水量
        '''
        if P_wf > GP.P_wf:
            q_w=GP.q_wi*30
        else:
            q_w=GP.get_water_prediction(P,k_w,P_wf,K)*30
        q_w_list.append(q_w/30)
        # print('q_w:', q_w/10)
        W_p = W_p + q_w
        '''
        计算含水饱和度
        '''
        S_w = GP.get_S_w(W_p,phi)
        S_w_list.append(S_w)
        # print('S_w:', S_w)

        '''
        计算压力
        '''
        if P > GP.P_cd:
            P=GP.get_P_1( S_w, Z, phi, G_p)
        else:
            P = GP.get_P_2(S_w, Z, phi, G_p)
        P_list.append(P)
        # print('P:',P)

        '''
        计算气体压缩因子
        '''
        Z = GP.get_z(P,GP.T,0.8)
        Z_list.append(Z)
        # print('Z:',Z)

        '''
        计算孔隙度
        '''
        phi=GP.get_phi(P)
        phi_list.append(phi)
        # print('phi:',phi)

        '''
        计算渗透率
        '''
        K=GP.get_K(phi)
        K_list.append(K)

        '''
        计算气水相渗透率
        '''
        k_g, k_w=GP.get_k_rg_k_rw(S_w)

        '''
        计算井底流压，若井底流压小于设定值，则认为当前井底流压为设定值
        '''
        if P_wf > GP.P_wf:
            # P_wf = GP.P_wf
            P_wf = GP.get_P_wf(P, k_w, K)
        else:
            P_wf = GP.P_wf
        P_wf_list.append(P_wf)

        '''
        根据当前压力，判断排采阶段，计算产气量
        '''
        if P > GP.P_cd:
            q_g=0
        else:
            q_g = GP.get_gas_prediction(P,k_g,Z,P_wf,K)*30
        q_g_list.append(q_g/30)

        G_p = G_p + q_g

    return G_p

def grid_gas_prediction(data_array,grid_info_in,range_x,range_y,compute_range_x,compute_range_y):
    grid_info=copy.deepcopy(grid_info_in)
    well_points_array = create_grid(grid_info, range_x, range_y,compute_range_x,compute_range_y)
    well_num = len(well_points_array)

    well_grid_GP = 0

    area = int(grid_info['x'] * grid_info['y'] * np.sin(grid_info['theta'] * np.pi / 180))

    for x_y in well_points_array:

        x = x_y[0]
        y = x_y[1]
        info_dict = get_cell_dict(data_array, range_x, range_y, 100, [x, y])

        well_info = {

            'A': area,
            'V_L': info_dict['V_L'],
            'P_L': info_dict['P_L'],
            'P_cd': info_dict['P_cd'],
            'P_i': info_dict['P_i'],
            'h': info_dict['h'],
            'phi_i': info_dict['phi_i'],
            'K_i': info_dict['K_i'],
            'rho_B': 1.58
        }
        G_p = well_gas_prediction(well_info, 180)

        well_grid_GP = well_grid_GP + G_p


    profit=well_grid_GP*1.44-6000000*well_num
    # profit = well_grid_GP /(well_num+0.00001)-150000

    # print('well_num:',well_num,' grid_GP:', well_grid_GP,' profit:',profit,' mean_well_profit:',profit/well_num,' mean_well_year_profit:',profit/(well_num*15))
    return profit


if __name__=="__main__":
    grid_info={
        'x':400,
        'y':400,
        'theta':90,
        'Delta_x':0,
        'Delta_y':0,
        'gamma':0
    }
    range_x=[623935,629935]
    range_y=[3959412,3965412]

    compute_range_x=[623935,623935+6000]
    compute_range_y=[3959412,3959412+6000]

    well_points_array=create_grid(grid_info,range_x,range_y,compute_range_x,compute_range_y)
    print('well num:',len(well_points_array))
    #
    # data_array=np.array(pd.read_csv('data/data_CBM_info.csv',header=None))

    data = np.array(pd.read_csv('data/IDW_解吸压力.csv', header=None))
    plt.imshow(data)
    # plt.colorbar(shrink=.83)

    draw_scatter(well_points_array,range_x,range_y)

    data_array = np.array(pd.read_csv('data/data_CBM_info.csv', header=None))
    profit=grid_gas_prediction(data_array,grid_info,range_x,range_y,compute_range_x,compute_range_y)
    print(profit)







