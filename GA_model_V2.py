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
        "mutate_rate":0.5,
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


        self.cell_len=200
        self.x_zoom_range=[10,31]
        self.y_zoom_range=[10,31]
        self.theta_range=[60,91]
        self.gamma_range=[0,91]
        self.Delta_x_range=[0,201]
        self.Delta_y_range=[0,201]

    def target_func(self,grid_info):
        x_range=self.data_info['x_range']
        y_range=self.data_info['y_range']
        data_cell_len=self.data_info['data_cell_len']
        sum_value,well_num = compute_sum_mean(self.data_array, x_range, y_range, data_cell_len, grid_info)
        return sum_value,well_num

    def create_pop(self):
        pop_ori=[]

        for num in range( self.GA_model["num_grids"]):
            cell_len=self.cell_len
            x_zoom=np.random.randint( self.x_zoom_range[0], self.x_zoom_range[1])
            y_zoom=np.random.randint( self.y_zoom_range[0], self.y_zoom_range[1])
            theta=np.random.randint( self.theta_range[0], self.theta_range[1])
            gamma=np.random.randint( self.gamma_range[0], self.gamma_range[1])
            Delta_x=np.random.randint( self.Delta_x_range[0], self.Delta_x_range[1])
            Delta_y=np.random.randint( self.Delta_y_range[0], self.Delta_y_range[1])
            grid_info={
                'cell_len':cell_len,
                'x_zoom':x_zoom,
                'y_zoom':y_zoom,
                'theta':theta,
                'gamma':gamma,
                'Delta_x':Delta_x,
                'Delta_y':Delta_y
            }
            pop_ori.append(grid_info)
        return pop_ori

    def compute(self,pop_ori):
        sum_list=[]
        for grid in pop_ori:
            sum_value, well_num=self.target_func(grid)
            sum_list.append(sum_value-well_num*self.data_info['well_price'])

        id_max = np.argmax(sum_list)
        print(np.mean(sum_list), sum_list[id_max], pop_ori[id_max])
        return sum_list

    # def compute_save(self, pop_save):
    #     sum_list = []
    #     for grid in pop_save:
    #         sum_value, well_num = self.target_func(grid)
    #         sum_list.append(sum_value - well_num * self.data_info['well_price'])
    #     id_max = np.argmax(sum_list)
    #     print(np.mean(sum_list), sum_list[id_max], pop_save[id_max])
    #     return np.mean(sum_list), sum_list[id_max]

    def select(self,sum_list,pop):
        grids_pop_copy=copy.deepcopy(pop)
        grids_pop_copy=np.array(grids_pop_copy)
        sum_list_copy=copy.deepcopy(sum_list)
        sum_list_copy=np.array(sum_list_copy)

        rate=self.GA_model['eliminate_rate']
        drop_num = int(self.GA_model['num_grids'] *rate)

        id_sort = sum_list_copy.argsort()

        save_id = id_sort[drop_num:self.GA_model['num_grids']]
        save_grids_pop_copy=grids_pop_copy[save_id]

        drop_id = id_sort[0:drop_num]
        drop_grids_pop_copy=grids_pop_copy[drop_id]
        drop_value = sum_list_copy[drop_id]

        fitness = drop_value - np.min(drop_value) + 1
        idx = np.random.choice(np.arange(len(drop_grids_pop_copy)), size=len(drop_grids_pop_copy), replace=True,
                               p=(fitness / np.sum(fitness)))

        drop_grids=drop_grids_pop_copy[idx]
        save_grids=save_grids_pop_copy

        return save_grids,drop_grids

    def crossover(self, drop_grids_pop_parent, save_grids_pop_parent):
        '''
        多点交叉
        :param drop_grids_pop_parent:
        :param save_grids_pop_parent:
        :return:
        '''
        drop_grids_pop_parent_copy=copy.deepcopy(drop_grids_pop_parent)
        save_grids_pop_parent_copy=list(copy.deepcopy(save_grids_pop_parent))

        for n in range(len(drop_grids_pop_parent_copy)):
            if np.random.rand() < self.GA_model['cross_rate']:
                i_save = np.random.randint(0,len(save_grids_pop_parent_copy))
                save_parent_grid = save_grids_pop_parent_copy[i_save]

                # i_a=int(np.random.randint(0,len(drop_grids_pop_parent_copy[n])))
                # i_b = int( np.random.randint(0,len(drop_grids_pop_parent_copy[n])))
                # range_min=min(i_a,i_b)
                # range_max=max(i_a, i_b)
                # num_cross=0

                for i in drop_grids_pop_parent_copy[n]:
                    if np.random.rand() < 0.5:
                        drop_grids_pop_parent_copy[n][i]=save_parent_grid[i]
                    # num_cross=num_cross+1


        drop_grids_pop_parent_copy_mutate=self.mutate(drop_grids_pop_parent_copy)


        save_grids_pop_parent_copy.extend(drop_grids_pop_parent_copy_mutate)

        return save_grids_pop_parent_copy


    # def crossover(self, drop_grids_pop_parent, save_grids_pop_parent):
    #     '''
    #     单点交叉
    #     '''
    #     drop_grids_pop_parent_copy=copy.deepcopy(drop_grids_pop_parent)
    #     save_grids_pop_parent_copy=list(copy.deepcopy(save_grids_pop_parent))
    #
    #     for n in range(len(drop_grids_pop_parent_copy)):
    #         if np.random.rand() < self.GA_model['cross_rate']:
    #             i_save = np.random.randint(0,len(save_grids_pop_parent_copy))
    #             save_parent_grid = save_grids_pop_parent_copy[i_save]
    #
    #             i_a=int(np.random.randint(0,len(drop_grids_pop_parent_copy[n])))
    #
    #             num_cross=0
    #
    #             for i in drop_grids_pop_parent_copy[n]:
    #                 if num_cross==i_a:
    #                     drop_grids_pop_parent_copy[n][i]=save_parent_grid[i]
    #                 num_cross=num_cross+1
    #
    #
    #     drop_grids_pop_parent_copy_mutate=self.mutate(drop_grids_pop_parent_copy)
    #
    #
    #     save_grids_pop_parent_copy.extend(drop_grids_pop_parent_copy_mutate)
    #
    #     return save_grids_pop_parent_copy

    def mutate(self,pop):

        dic_map={
            'x_zoom':self.x_zoom_range,
            'y_zoom':self.y_zoom_range,
            'theta':self.theta_range,
            'gamma': self.gamma_range,
            'Delta_x':self.Delta_x_range,
            'Delta_y':self.Delta_y_range
        }
        pop_copy=copy.deepcopy(pop)

        for DNA_line in pop_copy:
            if np.random.rand() < self.GA_model['mutate_rate']:
                id_mutate=np.random.randint(1,len(dic_map))
                name_mutate=list(dic_map.keys())[id_mutate]
                DNA_line[name_mutate]=np.random.randint(dic_map[name_mutate][0],dic_map[name_mutate][1])

        return pop_copy



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
        "num_grids":500,
        "num_iteration": 5000,
        "cross_rate":0.6,
        "mutate_rate":0.05,
        "eliminate_rate": 0.85
    }

    pop_ori=GA.create_pop()

    i_list = []
    mean_list = []
    max_list=[]

    for i in range(GA.GA_model['num_iteration']):
        print(i)
        sum_list=GA.compute(pop_ori)
        save_grids,drop_grids=GA.select(sum_list,pop_ori)
        i_list.append(i)

        # mean_list.append(result_mean)
        # max_list.append(result_max)

        pop_ori=GA.crossover( drop_grids, save_grids)
        # plt.scatter(i_list, mean_list, marker='o', color='red', s=40, label='First')
        # plt.scatter(i_list, max_list, marker='x', color='blue', s=40, label='First')
        # plt.show()




