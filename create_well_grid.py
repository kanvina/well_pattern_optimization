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

def draw_scatter(points_array,xlim,ylim):
    for point_location in points_array:


        plt.scatter(point_location[0], point_location[1], marker='o', color='red', s=5, label='First')
    plt.plot([xlim[0],xlim[1],xlim[1],xlim[0],xlim[0]],[ylim[0],ylim[0],ylim[1],ylim[1],ylim[0]])
        # plt.xlim(xlim)
        # plt.ylim(ylim)
    plt.show()

def create_grid(grid_info,range_x,range_y,compute_range_x,compute_range_y):

    grid=well_grid_class(grid_info)
    points_array=grid.create_rhombus_well_grid(range_x,range_y)

    well_points_array=[]

    for points_row in points_array:
        points_line = points_array[points_row]
        for point in points_line:
            point_LT = point[2]
            if point_LT[0]>=compute_range_x[0] and point_LT[0]<=compute_range_x[1] and point_LT[1] >=compute_range_y[0] and point_LT[1] <=compute_range_y[1]:
                well_points_array.append(point_LT)

    return well_points_array

def get_cell_value(data_array,x_range,y_range,cell_len,location):
    row=int( np.floor((y_range[1]-location[1])/cell_len))
    column=int(np.floor( (location[0]-x_range[0])/cell_len ))
    point_value=data_array[row,column]
    return point_value


if __name__=="__main__":
    grid_info={
        'x':300,
        'y':300,
        'theta':30,
        'Delta_x':100,
        'Delta_y':100,
        'gamma':45
    }
    range_x=[623935,629935]
    range_y=[3959412,3965412]

    compute_range_x=[624935,628935]
    compute_range_y=[3960412,3964412]

    well_points_array=create_grid(grid_info,range_x,range_y,compute_range_x,compute_range_y)
    #
    # data_array=np.array(pd.read_csv('data/data_CBM_info.csv',header=None))




    draw_scatter(well_points_array,range_x,range_y)







