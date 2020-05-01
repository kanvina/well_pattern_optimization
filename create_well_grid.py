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
from matplotlib.font_manager import FontProperties

class well_grid_class():

    def __init__(self,grid_info):

        self.x = grid_info['x']                  #横向间距
        self.y= grid_info['y']                   #纵向间距
        self.theta = grid_info['theta']          #单元夹角
        self.Delta_x = grid_info['Delta_x']      #横轴移动因子
        self.Delta_y = grid_info['Delta_y']      #纵轴移动因子
        self.gamma=grid_info['gamma']            #旋转角度

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
                angle = np.arcsin(np.abs(grid_center_point_location[1] - point[1]) / distance) * 180 / np.pi + 180
            # 第一象限
            elif point[1] - grid_center_point_location[1] >= 0 and point[0] - grid_center_point_location[0] >= 0:
                angle = np.arcsin(np.abs(point[1] - grid_center_point_location[1]) / distance) * 180 / np.pi
            # 第二象限
            elif point[1] - grid_center_point_location[1] >= 0 and point[0] - grid_center_point_location[0] <= 0:
                angle = 180 - np.arcsin(np.abs(point[1] - grid_center_point_location[1]) / distance) * 180 / np.pi
            # 第四象限
            elif point[1] - grid_center_point_location[1] <= 0 and point[0] - grid_center_point_location[0] >= 0:
                angle = 360-np.arcsin(np.abs(grid_center_point_location[1] - point[1]) / distance) * 180 / np.pi
        except:
            print(point, "点旋转出现异常")
        angle_new=angle- self.gamma
        if angle_new <0 :
            point_rotate_location = [distance * np.cos((-angle_new) * np.pi / 180) + grid_center_point_location[0],
                                     grid_center_point_location[1]-distance * np.sin((-angle_new )* np.pi / 180) ]
        elif angle_new>0 and angle_new <90:
            point_rotate_location = [distance * np.cos(angle_new * np.pi / 180) + grid_center_point_location[0],
                                 distance * np.sin(angle_new * np.pi / 180) + grid_center_point_location[1]]
        elif angle_new >90 and angle_new<180:
            angle_new=180-angle_new
            point_rotate_location = [  grid_center_point_location[0]-distance * np.cos(angle_new * np.pi / 180),
                                     distance * np.sin(angle_new * np.pi / 180) + grid_center_point_location[1]]
        elif angle_new >180 and angle_new<270:
            angle_new = angle_new-180
            point_rotate_location = [  grid_center_point_location[0]-distance * np.cos(angle_new * np.pi / 180),
                                     -distance * np.sin(angle_new * np.pi / 180) + grid_center_point_location[1]]
        else:
            point_rotate_location = [distance * np.cos((-angle_new) * np.pi / 180) + grid_center_point_location[0],
                                     grid_center_point_location[1] - distance * np.sin((-angle_new) * np.pi / 180)]


        return point_rotate_location

def draw_scatter(points_array,xlim,ylim,y_cell_num):

    for point_location in points_array:
        plt.scatter((point_location[0]-xlim[0])/100, (y_cell_num-(point_location[1]-ylim[0])/100), marker='o', color='red', s=2)
        # plt.pause(0.1)

    plt.xlim([0,(xlim[1]-xlim[0])/100])
    plt.ylim([(ylim[1]-ylim[0])/100,0])
    # plt.axis('off')
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    # plt.title(u'最佳布井方案变化',fontproperties=font_set)
    plt.xticks(np.linspace(0,60,4),[0,2,4,6])
    plt.yticks(np.linspace(0, 60, 4),[6,4,2,0])
    # plt.xlabel('km')
    # plt.ylabel('km')
    plt.show()


def create_grid(grid_info,range_x,range_y,compute_range_x,compute_range_y):

    grid=well_grid_class(grid_info)
    points_array=grid.create_rhombus_well_grid(range_x,range_y)

    well_points_array=[]

    for points_row in points_array:
        points_line = points_array[points_row]
        for point in points_line:
            point_LT = point[2]
            if point_LT[0] > compute_range_x[0]  and point_LT[0] < compute_range_x[1] and point_LT[1] >compute_range_y[0] and point_LT[1] < compute_range_y[1] :
            # if point_LT[0]>compute_range_x[0]+150 and point_LT[0]<compute_range_x[1]-150 and point_LT[1] >compute_range_y[0]+150 and point_LT[1] <compute_range_y[1]-150:

                well_points_array.append(point_LT)

    return well_points_array

def get_cell_value(data_array,x_range,y_range,cell_len,location):
    row=int( np.floor((y_range[1]-location[1])/cell_len))
    column=int(np.floor( (location[0]-x_range[0])/cell_len ))
    point_value=data_array[row,column]
    return point_value


def create_grid_show(grid_info, range_x, range_y, compute_range_x, compute_range_y,magin):
    grid = well_grid_class(grid_info)
    points_array = grid.create_rhombus_well_grid(range_x, range_y)

    well_points_array = []

    for points_row in points_array:
        points_line = points_array[points_row]
        for point in points_line:
            point_LT = point[2]

            if point_LT[0]>compute_range_x[0]+magin and point_LT[0]<compute_range_x[1]-magin and point_LT[1] >compute_range_y[0]+magin and point_LT[1] <compute_range_y[1]-magin:

                well_points_array.append(point_LT)

    return well_points_array

if __name__=="__main__":
    grid_info={'x': 350,
               'y': 250,
               'theta': 90,
               'gamma': 0,
               'Delta_x': 0,
               'Delta_y': 0}
    range_x=[623935,629935]
    range_y=[3959412,3965412]

    compute_range_x=[623935,623935+6000]
    compute_range_y=[3959412,3959412+6000]

    well_points_array=create_grid(grid_info,range_x,range_y,compute_range_x,compute_range_y)
    print('well num:',len(well_points_array))

    data = np.array(pd.read_csv('data/IDW_含气量.csv', header=None))
    plt.imshow(data)

    # cb=plt.colorbar(shrink=.83)
    # font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
    # cb.set_label('含气量（m³）',fontproperties=font_set)

    draw_scatter(well_points_array,range_x,range_y,60)










