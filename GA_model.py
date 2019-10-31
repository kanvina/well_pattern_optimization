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
        self.theta_range=[32,128]
        self.gamma_range=[0,128]
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

            print(grid_info)
            sum_num_mean_list=self.target_func(grid_info,self.data_array)
            print(sum_num_mean_list)

            pop_result_list.append(sum_num_mean_list[0])

        return pop_result_list

    def select(self,pop_result_list,grids_pop):  # 根据概率选择
        pop_result_list=np.array(pop_result_list)

        idx = np.random.choice(np.arange(self.GA_model['num_grids']), size=self.GA_model['num_grids'], replace=True,p=(pop_result_list / pop_result_list.sum()))

        grids_pop_parent=[]

        for i in idx:
            grids_pop_parent.append(grids_pop[i])

        return grids_pop_parent

    def crossover(self,grids_pop_parent):
        print(1)


if __name__=="__main__":

    GA=GA_model_grid()
    GA.data_info={
        'Path':'data/data_target.csv',
        'x_range':[615887, 635437],
        'y_range':[3946571, 3960751],
        'data_cell_len':100
    }
    GA.GA_model={
        "num_grids":20,
        "num_iteration": 5000,
        "cross_rate":0.7,
        "mutate_rate":0.005
    }

    grids_pop=GA.create_pop()
    pop_result_list=GA.compute_grids(grids_pop)


    for i in range(GA.GA_model['num_grids']):
        grids_pop_parent = GA.select(pop_result_list, grids_pop)
        pop_result_list = GA.compute_grids(grids_pop)
        grids_pop=grids_pop_parent


