import pandas as pd
import numpy as np
import copy
from matplotlib import pyplot as plt
from create_well_grid import create_grid
from gas_prediction import Gas_prediction
from matplotlib.font_manager import FontProperties

class GA():
    def __init__(self,GA_info):

        self.GA_range_x =GA_info['GA_range_x']
        self.GA_range_y = GA_info['GA_range_y']
        self.GA_range_theta =GA_info['GA_range_theta']
        self.GA_range_Delta_x= GA_info['GA_range_Delta_x']
        self.GA_range_Delta_y=GA_info['GA_range_Delta_y']
        self.GA_range_gamma=GA_info['GA_range_gamma']
        self.num=GA_info['num']
        self.n_max=GA_info['n_max']
        self.cross_rate=GA_info['cross_rate']
        self.mutate_rate=GA_info['mutate_rate']
        self.save_rate=GA_info['save_rate']

    def create_pop(self):
        pop_list=[]
        for i in range(self.num):
            grid_info = {
                'x': np.random.randint(self.GA_range_x[0],self.GA_range_x[1]),
                'y': np.random.randint(self.GA_range_y[0],self.GA_range_y[1]),
                'theta': np.random.randint(self.GA_range_theta[0],self.GA_range_theta[1]),
                'Delta_x': np.random.randint(self.GA_range_Delta_x[0],self.GA_range_Delta_x[1]),
                'Delta_y': np.random.randint(self.GA_range_Delta_y[0],self.GA_range_Delta_y[1]),
                'gamma': np.random.randint(self.GA_range_gamma[0],self.GA_range_gamma[1])
            }
            pop_list.append(grid_info)
        return pop_list

    def save(self,pop_list_in,fitness_list_in):
        fitness_list=copy.deepcopy(fitness_list_in)
        fitness_list=np.array(fitness_list)
        idx_list = np.argsort(fitness_list)[::-1]
        fitness_list_save=np.sort(fitness_list)[::-1][0:int(self.num*self.save_rate)]
        fitness_list_GA=np.sort(fitness_list)[::-1][int(self.num*self.save_rate)::]

        idx_save_list=idx_list[0:int(self.num*self.save_rate)]
        idx_GA_list=idx_list[int(self.num*self.save_rate)::]

        pop_list=copy.deepcopy(pop_list_in)

        pop_list=np.array(pop_list)

        pop_list_save=pop_list[idx_save_list]

        pop_list_GA=pop_list[idx_GA_list]

        return pop_list_GA,pop_list_save,fitness_list_GA,fitness_list_save

    def select(self, pop_list_GA,fitness_list_GA):  # 根据概率选择
        pop_len = len(fitness_list_GA)

        min_value = np.min(fitness_list_GA)
        for i in range(len(fitness_list_GA)):
            fitness_list_GA[i] = fitness_list_GA[i] - min_value + 1

        fitness_list = np.array(fitness_list_GA)

        idx = np.random.choice(np.arange(pop_len), size=pop_len, replace=True, p=(fitness_list / fitness_list.sum()))

        pop_list = np.array(pop_list_GA)

        pop_list_GA=pop_list[idx]
        fitness_GA_new_list = fitness_list_GA[idx]
        return pop_list_GA,fitness_GA_new_list

    '''
    多点
    '''
    def cross(self, pop_list_GA,fitness_list_GA,pop_list_save_in, n_now, n_max,data_array,range_x,range_y,compute_range_x,compute_range_y):
        fitness_list_child=[]

        # pop_list_save=copy.deepcopy(pop_list_save_in)
        pop_list_save = copy.deepcopy(pop_list_GA)
        a = (n_max - n_now) / n_max
        # a=0.5

        pop_copy_list = copy.deepcopy(pop_list_GA)
        pop_copy_len = len(pop_list_save)


        child_list = []
        i=-1

        for grid_info in pop_copy_list:
            grid_info_parent = copy.deepcopy(grid_info)

            i = i + 1
            fitness = fitness_list_GA[i]
            idx = np.random.randint(0, pop_copy_len)
            grid_info_copy = pop_list_save[idx]
            if np.random.rand() < self.cross_rate:

                if np.random.rand() < 0.5:
                    x=grid_info['x']
                    x_copy = grid_info_copy['x']
                    x_child = a * x_copy + (1 - a) * x
                    grid_info['x']=x_child

                elif np.random.rand() < 0.5:
                    y=grid_info['y']
                    y_copy = grid_info_copy['y']
                    y_child = a * y_copy + (1 - a) * y
                    grid_info['y']=y_child

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


            fitness_child = grid_gas_prediction(data_array, grid_info, range_x, range_y,compute_range_x,compute_range_y)

            if fitness_child >= fitness:
                child_list.append(grid_info)
                fitness_list_child.append(fitness_child)
            else:
                child_list.append(grid_info_parent)
                fitness_list_child.append(fitness)

        return child_list,fitness_list_child

    def mutate(self, child_list_in,fitness_list_child_in,GA_info,low_ratio,data_array,range_x, range_y, compute_range_x,compute_range_y):
        child_list=copy.deepcopy(child_list_in)
        fitness_list_child=copy.deepcopy(fitness_list_child_in)
        pop_list_mutate = []
        fitness_list_mutate=[]
        i=-1
        for grid_info in child_list:
            grid_info_parent=copy.deepcopy(grid_info)
            i=i+1
            num = len(grid_info)
            idx = np.random.randint(0, num)


            if np.random.rand() < self.mutate_rate:
                name = list(grid_info.keys())[idx]
                name_range=list(GA_info.keys())[idx]

                value_range=GA_info[name_range]
                grid_info[name]=np.random.randint(value_range[0],value_range[1])

                fitness_mutate = grid_gas_prediction(data_array, grid_info, range_x, range_y, compute_range_x,
                                                    compute_range_y)

                if fitness_mutate >=fitness_list_child[i]*(1-low_ratio):
                    pop_list_mutate.append(grid_info)
                    fitness_list_mutate.append(fitness_mutate)


                else:
                    pop_list_mutate.append(grid_info_parent)
                    fitness_list_mutate.append(fitness_list_child[i])
            else:
                pop_list_mutate.append(grid_info_parent)
                fitness_list_mutate.append(fitness_list_child[i])


        pop_list_mutate = np.array(pop_list_mutate)
        fitness_list_mutate=np.array(fitness_list_mutate)
        return pop_list_mutate,fitness_list_mutate

def get_cell_value(data_array,x_range,y_range,cell_len,location):
    row=int( np.floor((y_range[1]-location[1])/cell_len))
    column=int(np.floor( (location[0]-x_range[0])/cell_len ))
    point_value=data_array[row,column]
    info_dict = eval(point_value)
    return info_dict

# def well_gas_prediction(well_info,time):
#     G_p=well_info['A']
#     return G_p

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
        G_p = well_gas_prediction(well_info, 180)

        well_grid_GP = well_grid_GP + G_p


    profit=well_grid_GP*1.44-6000000*well_num
    # profit = well_grid_GP /(well_num+0.00001)

    # print('well_num:',well_num,' grid_GP:', well_grid_GP,' profit:',profit,' mean_well_profit:',profit/well_num,' mean_well_year_profit:',profit/(well_num*15))
    return profit

def get_fitness_list(pop_list,data_array,range_x,range_y,compute_range_x,compute_range_y):

    fitness_list=[]

    for grid_info in pop_list:
        profit = grid_gas_prediction(data_array, grid_info, range_x, range_y,compute_range_x,compute_range_y)
        fitness_list.append(profit)
    return fitness_list

if __name__=='__main__':

    range_x=[623935,629935]
    range_y=[3959412,3965412]

    # range_x=[623935,623935+500]
    # range_y=[3959412,3959412+500]

    compute_range_x=[623935,623935+2000]
    compute_range_y=[3959412,3959412+2000]

    data_array = np.array(pd.read_csv('data/data_CBM_info.csv', header=None))

    GA_info={

        'GA_range_x' :[300, 400],
        'GA_range_y' : [300, 400],
        'GA_range_theta' :[30, 90],
        'GA_range_Delta_x' : [0, 200],
        'GA_range_Delta_y' : [0, 200],
        'GA_range_gamma' : [0, 90],
        'num':50,
        'n_max':300,
        'cross_rate':0.8,
        'mutate_rate':0.1,
        'save_rate':0.15
    }
    GA = GA(GA_info)
    pop_list_all = GA.create_pop()
    fitness_list = get_fitness_list(pop_list_all, data_array, range_x, range_y, compute_range_x, compute_range_y)

    mean_list=[]
    max_list=[]
    i_list=[]

    for i in range(GA.n_max):
        i_list.append(i)

        mean_value=np.mean(fitness_list)
        max_value=np.max(fitness_list)
        mean_list.append(mean_value)
        max_list.append(max_value)
        idx=np.argmax(fitness_list)
        max_grid=pop_list_all[idx]

        # pd.DataFrame(zip(fitness_list,pop_list_all),columns=[i+1,'']).to_csv('data/GA_process.csv',mode='a+',index=0)
        print('Generation:',i+1,' mean_value:',int(mean_value),' max_value:',int(max_value),' max_grid:',max_grid)

        pop_list_GA, pop_list_save, fitness_list_GA, fitness_list_save=GA.save(pop_list_all,fitness_list)

        pop_list_GA, fitness_list_GA=GA.select(pop_list_GA,fitness_list_GA)

        pop_list_GA,fitness_list_GA = GA.cross(pop_list_GA,fitness_list_GA,pop_list_save, i, GA.n_max,data_array,range_x,range_y,compute_range_x,compute_range_y)

        pop_list_GA,fitness_list_GA=GA.mutate(pop_list_GA,fitness_list_GA,GA_info,0.1,data_array,range_x, range_y, compute_range_x,compute_range_y)

        pop_list_all = np.concatenate((pop_list_save, pop_list_GA), axis=0)
        fitness_list=np.concatenate((fitness_list_save, fitness_list_GA), axis=0)


    plt.plot(i_list, mean_list, marker='x', mec='blue',  lw=1,ms=5,label='种群平均值')
    plt.plot(i_list, max_list, marker='o', mec='red',  lw=1,ms=5,label='种群最优值')
    font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc")
    plt.title('改进遗传算法进行井网优化结果', fontproperties=font)
    plt.legend(prop=font)
    plt.show()











