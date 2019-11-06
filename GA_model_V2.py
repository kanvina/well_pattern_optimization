import numpy as np
import pandas as pd
import copy
from compute_grid_value import compute_sum_mean
from matplotlib import pyplot as plt

class GA_model_grid():

    def __init__(self):

        self.GA_model={
        "num_grids":50,
        "num_iteration": 5000,
        "cross_rate":0.7,
        "mutate_rate":0.005,
        "eliminate_rate":0.85
        }

        self.data_info={
        'Path':'data/data_target.csv',
        'x_range':[615887, 635437],
        'y_range':[3946571, 3960751],
        'data_cell_len':100,
        'well_price':8000000
        }

        self.data_array=np.array( pd.read_csv(self.data_info['Path'],header=None))

        self.cell_len_range=[2000,3000]
        self.x_zoom_range=[1,8]
        self.y_zoom_range=[1,8]
        self.theta_range=[30,90]
        self.gamma_range=[0,90]
        self.Delta_x_range=[0,512]
        self.Delta_y_range=[0,512]

    def target_func(self,grid_info):

        x_range=self.data_info['x_range']
        y_range=self.data_info['y_range']
        data_cell_len=self.data_info['data_cell_len']
        sum_value,well_num = compute_sum_mean(self.data_array, x_range, y_range, data_cell_len, grid_info)
        return sum_value,well_num

    def get_value_2(self,type):

        type_value_line=[]
        len=type[1]-type[0]

        i_num=0
        while 2**i_num<=len:
            i_num=i_num+1
        else:
            pass
        value_2 = list(np.random.randint(2, size=i_num))
        type_value_line.append(type[0])
        type_value_line.append(value_2)

        return type_value_line

    def translate_value_2(self,type_value_line):

        type_low=type_value_line[0]
        value_2=type_value_line[1]
        value_str=''
        for i in value_2:
            value_str=value_str+str(i)
        value=int(value_str,2)+type_low
        return value

    def create_DNA(self):
        DNA_line_value2=[]

        cell_len_value2=self.get_value_2(self.cell_len_range)
        DNA_line_value2.append(cell_len_value2)

        x_zoom_value2=self.get_value_2(self.x_zoom_range)
        DNA_line_value2.append(x_zoom_value2)

        y_zoom_value2=self.get_value_2(self.y_zoom_range)
        DNA_line_value2.append(y_zoom_value2)

        theta_value2=self.get_value_2(self.theta_range)
        DNA_line_value2.append(theta_value2)

        gamma_value2=self.get_value_2(self.gamma_range)
        DNA_line_value2.append(gamma_value2)

        Delta_x_value2=self.get_value_2(self.Delta_x_range)
        DNA_line_value2.append(Delta_x_value2)


        Delta_y_value2=self.get_value_2(self.Delta_y_range)
        DNA_line_value2.append(Delta_y_value2)

        return DNA_line_value2

    def create_pop(self):
        grids_pop=[]
        for i in range(self.GA_model['num_grids']):
            DNA_line_value2=self.create_DNA()
            grids_pop.append(DNA_line_value2)

        return grids_pop

    def compute_grids(self,grids_pop):

        pop_result_list=[]
        pop_result_grid_info=[]

        for DNA_line_value2 in grids_pop:

            cell_len=self.translate_value_2(DNA_line_value2[0])
            x_zoom=self.translate_value_2(DNA_line_value2[1])
            y_zoom=self.translate_value_2(DNA_line_value2[2])
            theta=self.translate_value_2(DNA_line_value2[3])
            gamma=self.translate_value_2(DNA_line_value2[4])
            Delta_x=self.translate_value_2(DNA_line_value2[5])
            Delta_y=self.translate_value_2(DNA_line_value2[6])

            grid_info = {
                'cell_len': cell_len,
                'x_zoom': x_zoom,
                'y_zoom': y_zoom,
                'theta': theta,
                'gamma': gamma,
                'Delta_x': Delta_x,
                'Delta_y': Delta_y
            }

            pop_result_grid_info.append(grid_info)
            sum_value,well_num=self.target_func(grid_info)
            pop_result_list.append(sum_value-well_num*self.data_info['well_price'])

        id_max=np.argmax(pop_result_list)
        max_grid=pop_result_grid_info[id_max]

        # print('均值：',np.mean(pop_result_list),'最大值：',pop_result_list[id_max],'最佳方案：',max_grid)

        return pop_result_list

    def compute_grids_save(self,save_grids_pop):
        grids_pop=copy.deepcopy(save_grids_pop)

        pop_result_list=[]
        pop_result_grid_info=[]

        for DNA_line_value2 in grids_pop:

            cell_len=self.translate_value_2(DNA_line_value2[0])
            x_zoom=self.translate_value_2(DNA_line_value2[1])
            y_zoom=self.translate_value_2(DNA_line_value2[2])
            theta=self.translate_value_2(DNA_line_value2[3])
            gamma=self.translate_value_2(DNA_line_value2[4])
            Delta_x=self.translate_value_2(DNA_line_value2[5])
            Delta_y=self.translate_value_2(DNA_line_value2[6])

            grid_info = {
                'cell_len': cell_len,
                'x_zoom': x_zoom,
                'y_zoom': y_zoom,
                'theta': theta,
                'gamma': gamma,
                'Delta_x': Delta_x,
                'Delta_y': Delta_y
            }
            pop_result_grid_info.append(grid_info)
            sum_value,well_num=self.target_func(grid_info)
            pop_result_list.append(sum_value-well_num*self.data_info['well_price'])
        id_max=np.argmax(pop_result_list)
        max_grid=pop_result_grid_info[id_max]

        print('均值：',np.mean(pop_result_list),'最大值：',pop_result_list[id_max],'最佳方案：',max_grid)

        return np.mean(pop_result_list),pop_result_list[id_max]

    def select(self,pop_result_list,grids_pop):  # 根据概率选择
        grids_pop_copy=grids_pop
        pop_result_list=np.array(pop_result_list)
        rate=self.GA_model['eliminate_rate']
        drop_num = int(self.GA_model['num_grids'] *rate)

        id_sort = pop_result_list.argsort()

        save_id = id_sort[drop_num:self.GA_model['num_grids']]
        save_grids_pop_parent=[]
        for i in save_id:
            save_grids_pop_parent.append(grids_pop_copy[i])

        drop_id = id_sort[0:drop_num]
        drop_grids_pop_parent = []
        for n in drop_id:
            drop_grids_pop_parent.append(grids_pop_copy[n])

        drop_value=pop_result_list[drop_id]
        fitness=drop_value-np.min(drop_value)
        idx = np.random.choice(np.arange(len(drop_grids_pop_parent)), size=len(drop_grids_pop_parent), replace=True,p=(fitness / fitness.sum()))

        drop_grids=[]
        for m in idx:
            drop_grids.append(drop_grids_pop_parent[m])

        return drop_grids,save_grids_pop_parent

    def crossover(self,drop_grids_pop_parent,save_grids_pop_parent):
        drop_child_list=[]
        save_grids_pop=copy.deepcopy(save_grids_pop_parent)


        # for drop_line in drop_grids_pop_parent:
        #     print(1)
        #
        #     if np.random.rand() < self.GA_model['cross_rate']:
        #         i_pop_save = np.random.randint(0,len(save_grids_pop))
        #         grids_pop_line_save = save_grids_pop[i_pop_save]
        #     # else:
        #



        for n in range(len(drop_grids_pop_parent)):
            if np.random.rand() < self.GA_model['cross_rate']:

                i_pop_save = np.random.randint(0,len(save_grids_pop))
                grids_pop_line_save = save_grids_pop[i_pop_save]

                for i in range(len(drop_grids_pop_parent[n])):
                    grid_child = []

                    grid_DNA_parent = drop_grids_pop_parent[n][i][1]
                    grid_DNA_len = len(grid_DNA_parent)

                    i_cross = np.random.randint(0,grid_DNA_len)

                    grid_DNA_save = grids_pop_line_save[i][1]


                    grid_DNA_save_cross = grid_DNA_save[0:i_cross]
                    grid_DNA_parent_cross = grid_DNA_parent[i_cross:grid_DNA_len]


                    grid_child.extend(grid_DNA_save_cross)
                    grid_child.extend(grid_DNA_parent_cross)

                    drop_grids_pop_parent[n][i][1] = grid_child

        drop_grids_pop_parent_mutate=self.mutate(drop_grids_pop_parent)
        # drop_grids_pop_parent_mutate=drop_grids_pop_parent

        save_grids_pop_parent.extend(drop_grids_pop_parent_mutate)

        girds_pop_child=save_grids_pop_parent
        return girds_pop_child

    # def crossover(self,grids_pop_parent):
    #     pop_copy=grids_pop_parent.copy()
    #
    #     for n in range(len(grids_pop_parent)):
    #
    #         if np.random.rand() < self.GA_model['cross_rate']:
    #             i_pop = np.random.randint(0, self.GA_model["num_grids"])
    #             grids_pop_line=pop_copy[i_pop]
    #             for i in range(len(grids_pop_parent[n])):
    #                 grid_child=[]
    #                 grid_DNA_parent=grids_pop_parent[n][i][1]
    #                 grid_DNA_len=len(grid_DNA_parent)
    #                 grid_DNA_pop=grids_pop_line[i][1]
    #                 i_cross=np.random.randint(0,grid_DNA_len)
    #                 grid_DNA_parent_cross=grid_DNA_parent[0:i_cross]
    #                 grid_DNA_pop_cross=grid_DNA_pop[i_cross:grid_DNA_len]
    #                 grid_child.extend(grid_DNA_parent_cross)
    #                 grid_child.extend(grid_DNA_pop_cross)
    #                 grids_pop_parent[n][i][1]=grid_child
    #     return grids_pop_parent

    # def crossover(self, grids_pop_parent):
    #     pop_copy=grids_pop_parent.copy()
    #
    #     for n in range(len(grids_pop_parent)):
    #         if np.random.rand() < self.GA_model['cross_rate']:
    #             i_pop = np.random.randint(0, self.GA_model["num_grids"])
    #             grids_pop_line = pop_copy[i_pop]
    #             for i in range(len(grids_pop_parent[n])):
    #                 grid_child = []
    #                 grid_DNA_parent = grids_pop_parent[n][i][1]
    #                 grid_DNA_len = len(grid_DNA_parent)
    #                 grid_DNA_pop = grids_pop_line[i][1]
    #                 i_cross = np.random.randint(0, grid_DNA_len)
    #                 grid_DNA_parent_cross = grid_DNA_parent[0:i_cross]
    #                 grid_DNA_pop_cross = grid_DNA_pop[i_cross:grid_DNA_len]
    #                 grid_child.extend(grid_DNA_parent_cross)
    #                 grid_child.extend(grid_DNA_pop_cross)
    #                 grids_pop_parent[n][i][1] = grid_child
    #     return grids_pop_parent

    def mutate(self,drop_grids_pop_parent):

        girds_pop_mutate=copy.deepcopy(drop_grids_pop_parent)

        for i in range(len(girds_pop_mutate)):
            child_DNA_line=girds_pop_mutate[i]
            for n in range(len(child_DNA_line)):
                DNA=child_DNA_line[n][1]
                for id_mutate in range(len(DNA)):
                    if np.random.rand() < self.GA_model['mutate_rate']:
                        if DNA[id_mutate]==1:
                            DNA[id_mutate]=0
                        else:
                            DNA[id_mutate]=1
        return girds_pop_mutate


if __name__=="__main__":

    GA=GA_model_grid()
    GA.data_info={
        'Path':'data/data_target.csv',
        'x_range':[615887, 635437],
        'y_range':[3946571, 3960751],
        'data_cell_len':100,
        'well_price': 3920800
    }

    GA.GA_model={
        "num_grids":100,
        "num_iteration": 1500,
        "cross_rate":0.85,
        "mutate_rate":0.003,
        "eliminate_rate": 0.70
    }

    grids_pop=GA.create_pop()

    i_list = []
    mean_list = []
    max_list=[]

    for i in range(GA.GA_model['num_iteration']):


        pop_result_list = GA.compute_grids(grids_pop)
        drop_grids_pop_parent,save_grids_pop_parent = GA.select(pop_result_list,grids_pop)


        print(1+i,'save')
        mean,max=GA.compute_grids_save(save_grids_pop_parent)
        i_list.append(i)
        mean_list.append(mean)
        max_list.append(max)
        plt.scatter(i_list, mean_list, marker='o', color='red', s=40, label='First')
        plt.scatter(i_list, max_list, marker='x', color='blue', s=40, label='First')
        plt.pause(0.05)


        child=GA.crossover(drop_grids_pop_parent,save_grids_pop_parent)
        grids_pop=child



