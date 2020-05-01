import pandas as pd
import numpy as np
import copy
from matplotlib import pyplot as plt
from create_well_grid import create_grid
from gas_prediction import Gas_prediction
from matplotlib.font_manager import FontProperties
import create_well_grid as cwg


class GA():
    def __init__(self, GA_info):
        '''
        初始化
        :param GA_info: 各种变化参数，以字典形式存在。
        '''
        self.GA_range_x = GA_info['GA_range_x']  # 横向间距变化范围
        self.GA_range_y = GA_info['GA_range_y']  # 纵向间距变化范围
        self.GA_range_theta = GA_info['GA_range_theta']  # 井网单元夹角变化范围
        self.GA_range_Delta_x = GA_info['GA_range_Delta_x']  # 横向平移距离变化范围
        self.GA_range_Delta_y = GA_info['GA_range_Delta_y']  # 纵向平移距离变化范围
        self.GA_range_gamma = GA_info['GA_range_gamma']  # 旋转角度变化范围
        self.num = GA_info['num']  # 种群数目
        self.n_max = GA_info['n_max']  # 最大迭代次数
        self.cross_rate = GA_info['cross_rate']  # 交叉因子
        self.mutate_rate = GA_info['mutate_rate']  # 变异因子
        self.save_rate = GA_info['save_rate']  # 保留因子

    def create_pop(self):
        '''
        个体编码与种群生成
        :return: 种群
        '''
        pop_list = []
        for i in range(self.num):
            grid_info = {
                'x': np.random.randint(self.GA_range_x[0], self.GA_range_x[1]),
                'y': np.random.randint(self.GA_range_y[0], self.GA_range_y[1]),
                'theta': np.random.randint(self.GA_range_theta[0], self.GA_range_theta[1]),
                'Delta_x': np.random.randint(self.GA_range_Delta_x[0], self.GA_range_Delta_x[1]),
                'Delta_y': np.random.randint(self.GA_range_Delta_y[0], self.GA_range_Delta_y[1]),
                'gamma': np.random.randint(self.GA_range_gamma[0], self.GA_range_gamma[1])
            }
            pop_list.append(grid_info)
        return pop_list

    def save(self, pop_list_in, fitness_list_in):
        '''
        保留最优策略
        :param pop_list_in: 种群
        :param fitness_list_in: 种群个体适应度
        :return:
        pop_list_GA:非保留种群
        pop_list_save:保留种群
        fitness_list_GA:非保留种群适应度
        fitness_list_save:保留种群适应度
        '''
        fitness_list = copy.deepcopy(fitness_list_in)
        fitness_list = np.array(fitness_list)
        idx_list = np.argsort(fitness_list)[::-1]
        fitness_list_save = np.sort(fitness_list)[::-1][0:int(self.num * self.save_rate)]
        fitness_list_GA = np.sort(fitness_list)[::-1][int(self.num * self.save_rate)::]

        idx_save_list = idx_list[0:int(self.num * self.save_rate)]
        idx_GA_list = idx_list[int(self.num * self.save_rate)::]

        pop_list = copy.deepcopy(pop_list_in)

        pop_list = np.array(pop_list)

        pop_list_save = pop_list[idx_save_list]

        pop_list_GA = pop_list[idx_GA_list]

        return pop_list_GA, pop_list_save, fitness_list_GA, fitness_list_save

    def select(self, pop_list_GA, fitness_list_GA):
        '''
        选择操作
        :param pop_list_GA: 输入种群
        :param fitness_list_GA: 输入种群个体适应度
        :return:
        pop_list_GA:选择操作后种群
        fitness_GA_new_list:选择操作后种群个体适应度
        '''
        pop_len = len(fitness_list_GA)
        min_value = np.min(fitness_list_GA)
        for i in range(len(fitness_list_GA)):
            fitness_list_GA[i] = fitness_list_GA[i] - min_value + 0.0001
        fitness_list = np.array(fitness_list_GA)
        idx = np.random.choice(np.arange(pop_len), size=pop_len, replace=True, p=(fitness_list / fitness_list.sum()))
        for i in range(len(fitness_list_GA)):
            fitness_list_GA[i] = fitness_list_GA[i] + min_value - 1
        pop_list = np.array(pop_list_GA)
        pop_list_GA = pop_list[idx]
        fitness_GA_new_list = fitness_list_GA[idx]
        return pop_list_GA, fitness_GA_new_list

    '''
    多点
    '''

    def cross(self, pop_list_GA, fitness_list_GA, n_now, n_max, data_array, range_x, range_y, compute_range_x,
              compute_range_y):
        '''
        多点交叉操作，主要参数如下：
        :param pop_list_GA: 输入种群
        :param fitness_list_GA: 输入种群个体适应度
        :param n_now: 当前迭代次数
        :param n_max: 最大迭代次数
        :return:
        child_list:交叉操作后新种群
        fitness_list_child:交叉操作后新种群个体适应度
        '''
        fitness_list_child = []
        pop_list_save = copy.deepcopy(pop_list_GA)
        a = (n_max - n_now) / n_max
        pop_copy_list = copy.deepcopy(pop_list_GA)
        pop_copy_len = len(pop_list_save)
        child_list = []
        i = -1

        for grid_info in pop_copy_list:
            grid_info_parent = copy.deepcopy(grid_info)

            i = i + 1
            fitness = fitness_list_GA[i]
            idx = np.random.randint(0, pop_copy_len)
            grid_info_copy = pop_list_save[idx]
            if np.random.rand() < self.cross_rate:

                if np.random.rand() < 0.5:
                    x = grid_info['x']
                    x_copy = grid_info_copy['x']
                    x_child = a * x_copy + (1 - a) * x
                    grid_info['x'] = x_child

                elif np.random.rand() < 0.5:
                    y = grid_info['y']
                    y_copy = grid_info_copy['y']
                    y_child = a * y_copy + (1 - a) * y
                    grid_info['y'] = y_child

                elif np.random.rand() < 0.5:
                    theta = grid_info['theta']
                    theta_copy = grid_info_copy['theta']
                    theta_child = a * theta_copy + (1 - a) * theta
                    grid_info['theta'] = theta_child

                elif np.random.rand() < 0.5:
                    Delta_x = grid_info['Delta_x']
                    Delta_x_copy = grid_info_copy['Delta_x']
                    Delta_x_child = a * Delta_x_copy + (1 - a) * Delta_x
                    grid_info['Delta_x'] = Delta_x_child

                elif np.random.rand() < 0.5:
                    Delta_y = grid_info['Delta_y']
                    Delta_y_copy = grid_info_copy['Delta_y']
                    Delta_y_child = a * Delta_y_copy + (1 - a) * Delta_y
                    grid_info['Delta_y'] = Delta_y_child

                elif np.random.rand() < 0.5:
                    gamma = grid_info['gamma']
                    gamma_copy = grid_info_copy['gamma']
                    gamma_child = a * gamma_copy + (1 - a) * gamma
                    grid_info['gamma'] = gamma_child

            fitness_child = grid_gas_prediction(data_array, grid_info, range_x, range_y, compute_range_x,
                                                compute_range_y)

            if fitness_child >= fitness:
                child_list.append(grid_info)
                fitness_list_child.append(fitness_child)
            else:
                child_list.append(grid_info_parent)
                fitness_list_child.append(fitness)

        return child_list, fitness_list_child

    def mutate(self, child_list_in, fitness_list_child_in, GA_info, low_ratio, data_array, range_x, range_y,
               compute_range_x, compute_range_y):
        '''
        变异操作，主要参数如下：
        :param child_list_in: 输入种群
        :param fitness_list_child_in: 输入种群个体适应度
        :return:
        pop_list_mutate:变异操作后种群
        fitness_list_mutate:变异操作后种群个体适应度
        '''
        child_list = copy.deepcopy(child_list_in)
        fitness_list_child = copy.deepcopy(fitness_list_child_in)
        pop_list_mutate = []
        fitness_list_mutate = []
        i = -1
        for grid_info in child_list:
            grid_info_parent = copy.deepcopy(grid_info)
            i = i + 1
            num = len(grid_info)
            idx = np.random.randint(0, num)

            if np.random.rand() < self.mutate_rate:
                name = list(grid_info.keys())[idx]
                name_range = list(GA_info.keys())[idx]

                value_range = GA_info[name_range]
                grid_info[name] = np.random.randint(value_range[0], value_range[1])

                fitness_mutate = grid_gas_prediction(data_array, grid_info, range_x, range_y, compute_range_x,
                                                     compute_range_y)

                if fitness_mutate - fitness_list_child[i] >= 0 or (fitness_list_child[i] - fitness_mutate) >= np.abs(
                        (1 - low_ratio) * fitness_list_child[i]):
                    pop_list_mutate.append(grid_info)
                    fitness_list_mutate.append(fitness_mutate)

                else:
                    pop_list_mutate.append(grid_info_parent)
                    fitness_list_mutate.append(fitness_list_child[i])
            else:
                pop_list_mutate.append(grid_info_parent)
                fitness_list_mutate.append(fitness_list_child[i])

        pop_list_mutate = np.array(pop_list_mutate)
        fitness_list_mutate = np.array(fitness_list_mutate)
        return pop_list_mutate, fitness_list_mutate


def get_cell_value(data_array, x_range, y_range, cell_len, location):
    row = int(np.floor((y_range[1] - location[1]) / cell_len))
    column = int(np.floor((location[0] - x_range[0]) / cell_len))
    point_value = data_array[row, column]
    info_dict = eval(point_value)
    return info_dict

def well_gas_prediction(well_info, time):
    GP = Gas_prediction(well_info)

    '''
    定义列表，存放结果
    '''
    q_g_list = []
    q_w_list = []
    P_list = []
    i_list = []
    Z_list = []
    phi_list = []
    S_w_list = []
    G_P_list = []
    P_wf_list = []
    K_list = []
    Kg_list = []

    '''
    定义初始参数
    '''
    P = GP.P_i
    W_p = 0
    G_p = 0
    Z = GP.Z_i
    phi = GP.phi_i
    K = GP.K_i
    S_w = GP.S_wi
    k_g, k_w = GP.get_k_rg_k_rw(S_w)
    P_wf = GP.get_P_wf(P, k_w, K)
    '''
    设定排采时间，动态预测
    '''
    profit_well = 0
    for i in range(time):
        # print(i+1)
        i_list.append(i * 30)
        '''
        排水，根据井底流压计算排水量
        '''
        if P_wf > GP.P_wf:
            q_w = GP.q_wi * 30
        else:
            q_w = GP.get_water_prediction(P, k_w, P_wf, K) * 30
        q_w_list.append(q_w / 30)
        # print('q_w:', q_w/10)
        W_p = W_p + q_w
        '''
        计算含水饱和度
        '''
        S_w = GP.get_S_w(W_p, phi)
        S_w_list.append(S_w)
        # print('S_w:', S_w)

        '''
        计算压力
        '''
        if P > GP.P_cd:
            P = GP.get_P_1(S_w, Z, phi, G_p)
        else:
            P = GP.get_P_2(S_w, Z, phi, G_p)
        P_list.append(P)
        # print('P:',P)

        '''
        计算气体压缩因子
        '''
        Z = GP.get_z(P, GP.T, 0.8)
        Z_list.append(Z)
        # print('Z:',Z)

        '''
        计算孔隙度
        '''
        phi = GP.get_phi(P)
        phi_list.append(phi)
        # print('phi:',phi)

        '''
        计算渗透率
        '''
        K = GP.get_K(phi)
        K_list.append(K)

        '''
        计算气水相渗透率
        '''
        k_g, k_w = GP.get_k_rg_k_rw(S_w)

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
            q_g = 0
        else:
            q_g = GP.get_gas_prediction(P, k_g, Z, P_wf, K) * 30
        q_g_list.append(q_g / 30)

        G_p = G_p + q_g

        t = np.floor(i / 12) + 1

        profit = (q_g * 1.44 - 400000 / 12) * (1) ** (- t)
        profit_well = profit_well + profit

    return profit_well

def grid_gas_prediction(data_array, grid_info_in, range_x, range_y, compute_range_x, compute_range_y):
    grid_info = copy.deepcopy(grid_info_in)
    well_points_array = create_grid(grid_info, range_x, range_y, compute_range_x, compute_range_y)
    well_num = len(well_points_array)

    well_grid_GP = 0

    area = int(grid_info['x'] * grid_info['y'] * np.sin(grid_info['theta'] * np.pi / 180))
    profit_grid = 0

    for x_y in well_points_array:
        x = x_y[0]
        y = x_y[1]
        info_dict = get_cell_value(data_array, range_x, range_y, 100, [x, y])

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
        profit_well = well_gas_prediction(well_info, 180)

        profit_grid = profit_grid + profit_well

    return profit_grid,well_num




if __name__ == '__main__':

    range_x = [623935, 629935]
    range_y = [3959412, 3965412]


    compute_range_x = [623935, 623935 + 1000]
    compute_range_y = [3959412, 3959412 + 1000]

    data_array = np.array(pd.read_csv('data/data_CBM_info.csv', header=None))

    grid_info={'x': 306,
               'y': 328,
               'theta': 77,
               'gamma': 2,
               'Delta_x': 189,
               'Delta_y': 183}
    profit,well_num = grid_gas_prediction(data_array, grid_info, range_x, range_y, compute_range_x, compute_range_y)

    compute_range_x_show=[623935,623935+6000]
    compute_range_y_show=[3959412,3959412+6000]

    well_points_array_show_A=cwg.create_grid(grid_info,range_x,range_y,compute_range_x_show,compute_range_y_show)
    ori_num=len(well_points_array_show_A)

    well_points_array_show=cwg.create_grid_show(grid_info,range_x,range_y,compute_range_x_show,compute_range_y_show,0.5*min(grid_info['x'],grid_info['y']))

    new_num=len(well_points_array_show)
    print((profit*23/10000))

    print('num: ',new_num,' profit: ',(profit*23/10000)*new_num/ori_num)

    data = np.array(pd.read_csv('data/IDW_含气量.csv', header=None))
    plt.imshow(data)

    # cb=plt.colorbar(shrink=.83)
    # font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    # cb.set_label('含气量（m³/t）',fontproperties=font_set)

    cwg.draw_scatter(well_points_array_show, range_x, range_y, 60)


