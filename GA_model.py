import numpy as np
import pandas as pd
from compute_grid_value import compute_sum_mean

class GA_model_grid():

    def __init__(self):

        self.GA_model={
        "num_grids":50,
        "num_iteration": 5000,
        "cross_rate":0.7,
        "mutate_rate":0.005
        }

        self.data_info={
        'Path':'data/data_target.csv',
        'x_range':[615887, 635437],
        'y_range':[3946571, 3960751],
        'data_cell_len':100
        }

        self.data_array=np.array( pd.read_csv(self.data_info['Path'],header=None))

        self.cell_len_range=[64,128]
        self.x_zoom_range=[2,16]
        self.y_zoom_range=[2,16]
        self.theta_range=[30,90]
        self.gamma_range=[0,90]
        self.Delta_x_range=[0,512]
        self.Delta_y_range=[0,512]

    def target_func(self,grid_info,data_array):

        x_range=self.data_info['x_range']
        y_range=self.data_info['y_range']
        data_cell_len=self.data_info['data_cell_len']
        sum_num_mean_list = compute_sum_mean(data_array, x_range, y_range, data_cell_len, grid_info)
        return sum_num_mean_list

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

            # print(grid_info)
            pop_result_grid_info.append(grid_info)
            sum_num_mean_list=self.target_func(grid_info,self.data_array)
            pop_result_list.append(sum_num_mean_list[0])

        id_max=np.argmax(pop_result_list)
        max_grid=pop_result_grid_info[id_max]



        print('均值：',np.mean(pop_result_list),'最大值：',pop_result_list[id_max],'最佳方案：',max_grid)


        return pop_result_list

    def select(self,pop_result_list,grids_pop):  # 根据概率选择


        pop_result_list=np.array(pop_result_list)
        rate=0.8
        drop_num = int(self.GA_model['num_grids'] *rate)
        id_sort = pop_result_list.argsort()


        save_id = id_sort[drop_num:self.GA_model['num_grids']]
        save_grids_pop_parent=[]
        for i in save_id:
            save_grids_pop_parent.append(grids_pop[i])

        drop_id = id_sort[0:drop_num]

        drop_grids_pop_parent = []
        for i in drop_id:
            drop_grids_pop_parent.append(grids_pop[i])

        drop_value=pop_result_list[drop_id]
        idx = np.random.choice(np.arange(len(drop_grids_pop_parent)), size=len(drop_grids_pop_parent), replace=True,p=(drop_value / drop_value.sum()))

        drop=[]
        for i in idx:
            drop.append(drop_grids_pop_parent[i])

        return drop,save_grids_pop_parent

    def crossover(self,drop_grids_pop_parent,save_grids_pop_parent):
        # grids_pop_parent=[]

        for n in range(len(drop_grids_pop_parent)):
            if np.random.rand() < self.GA_model['cross_rate']:
                i_pop = np.random.randint(0,len(save_grids_pop_parent))
                grids_pop_line = save_grids_pop_parent[i_pop]
                for i in range(len(drop_grids_pop_parent[n])):
                    grid_child = []
                    grid_DNA_parent = drop_grids_pop_parent[n][i][1]
                    grid_DNA_len = len(grid_DNA_parent)
                    grid_DNA_pop = grids_pop_line[i][1]
                    i_cross = np.random.randint(0, grid_DNA_len)
                    grid_DNA_parent_cross = grid_DNA_parent[0:i_cross]
                    grid_DNA_pop_cross = grid_DNA_pop[i_cross:grid_DNA_len]

                    grid_child.extend(grid_DNA_parent_cross)
                    grid_child.extend(grid_DNA_pop_cross)
                    drop_grids_pop_parent[n][i][1] = grid_child

        drop_grids_pop_parent_mutate=self.mutate(drop_grids_pop_parent)
        drop_grids_pop_parent_mutate.extend(save_grids_pop_parent)


        girds_pop_child=drop_grids_pop_parent_mutate
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

    def mutate(self,girds_pop_child):
        for i in range(len(girds_pop_child)):
            child_DNA_line=girds_pop_child[i]
            if np.random.rand() < self.GA_model['mutate_rate']:
                for n in range(len(child_DNA_line)):
                    DNA=child_DNA_line[n][1]
                    id_mutate= np.random.randint(0,len(DNA))

                    if DNA[id_mutate]==1:
                        DNA[id_mutate]=0
                    else:
                        DNA[id_mutate]=1
        return girds_pop_child


if __name__=="__main__":

    GA=GA_model_grid()
    GA.data_info={
        'Path':'data/data_target.csv',
        'x_range':[615887, 635437],
        'y_range':[3946571, 3960751],
        'data_cell_len':100
    }
    GA.GA_model={
        "num_grids":10,
        "num_iteration": 150,
        "cross_rate":0.8,
        "mutate_rate":0.01
    }

    grids_pop=GA.create_pop()


    for i in range(GA.GA_model['num_iteration']-1):
        print(1+i)
        pop_result_list = GA.compute_grids(grids_pop)
        drop_grids_pop_parent,save_grids_pop_parent = GA.select(pop_result_list, grids_pop)
        girds_pop_child=GA.crossover(drop_grids_pop_parent,save_grids_pop_parent)
        # girds_pop_child=GA.mutate(girds_pop_child)
        grids_pop=girds_pop_child



