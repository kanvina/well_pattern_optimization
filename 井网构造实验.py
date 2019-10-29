'''
create by WYD
2019 10 25
井网构造实验
1- 构造原始井网单元
2- 横向与纵向缩放，因子：x_zoom,y_zoom
3- 横向与纵向平移：因子：Delta_x,Delta_y
4- 井网单元形状改变，因子：夹角theta
5- 井网单元旋转，因子：gamma
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class well_grid_class():

    def __init__(self):

        self.cell_len = 100  #原始井组边长
        self.x_zoom = 1      #横轴缩放因子
        self.y_zoom = 1      #纵轴缩放因子
        self.theta = 90      #井组夹角
        self.gamma = 45      #旋转角度
        self.Delta_x = 50    #横轴移动因子
        self.Delta_y = 50    #纵轴移动因子

    # def init_five_cell(self):
    #
    #     x_cell_len=self.cell_len*self.x_zoom
    #     y_cell_len=self.cell_len*self.y_zoom
    #
    #     angle_a= (np.arctan( (np.cos(self.theta*np.pi/180)+(y_cell_len/x_cell_len) )/np.sin(self.theta*np.pi/180)))*180/np.pi
    #     angle_b=180-self.theta-angle_a
    #     angle_c=angle_a-90+self.theta-self.gamma
    #     diagonal_line_len = y_cell_len*np.sin(angle_a*np.pi/180)+x_cell_len*np.sin(angle_b*np.pi/180)
    #
    #     point_LD_location=[self.center_location[0]-(self.cell_len/2)+self.Delta_x,self.center_location[1]-(self.cell_len/2)+self.Delta_y]
    #     point_LT_location=[y_cell_len*np.cos((self.theta-self.gamma)*np.pi/180)+point_LD_location[0],y_cell_len*np.sin((self.theta-self.gamma)*np.pi/180)+point_LD_location[1]]
    #     point_RT_location=[ diagonal_line_len*np.cos(angle_c*np.pi/180)+point_LD_location[0],diagonal_line_len*np.sin(angle_c*np.pi/180)+point_LD_location[1]  ]
    #     point_RD_location=[ x_cell_len*np.cos(self.gamma*np.pi/180)+point_LD_location[0],point_LD_location[1]-x_cell_len*np.sin(self.gamma*np.pi/180) ]
    #     point_center_location=[ 0.5*diagonal_line_len*np.cos((angle_a+self.theta-self.gamma-90)*np.pi/180)+point_LD_location[0],0.5*diagonal_line_len*np.sin((angle_a+self.theta-self.gamma-90)*np.pi/180)+point_LD_location[1] ]
    #
    #     return [point_LD_location,point_LT_location,point_RT_location,point_RD_location,point_center_location]

    def grid_rotate(self,grid_center_point_location, point):
        '''
        井网旋转函数
        :param grid_center_point_location: 旋转中心点
        :param point: 待旋转点
        :return:
        '''
        #待旋转点与中心点距离
        distance = ((point[1] - grid_center_point_location[1]) ** 2 + (
                    point[0] - grid_center_point_location[0]) ** 2) ** 0.5
        if distance==0:
            distance=1
        try:
            # 第三象限
            if point[1] - grid_center_point_location[1]<=0 and point[0] - grid_center_point_location[0]<=0:
                angle= np.arcsin(np.abs(point[1] - grid_center_point_location[1])/distance)* 180 / np.pi+180
            # 第一象限
            elif point[1] - grid_center_point_location[1]>=0 and point[0] - grid_center_point_location[0]>=0:
                angle= np.arcsin(np.abs(point[1] - grid_center_point_location[1])/distance)* 180 / np.pi
            # 第二象限
            elif point[1] - grid_center_point_location[1]>=0 and point[0] - grid_center_point_location[0]<=0:
                angle= 180-np.arcsin(np.abs(point[1] - grid_center_point_location[1])/distance)* 180 / np.pi
            # 第四象限
            elif point[1] - grid_center_point_location[1]<=0 and point[0] - grid_center_point_location[0]>=0:
                angle= -np.arcsin(np.abs(point[1] - grid_center_point_location[1])/distance)* 180 / np.pi
        except:
            print(point,"点旋转出现异常")
        point_rotate_location = [distance * np.cos((angle -self.gamma)*np.pi/180)+grid_center_point_location[0],
                                 distance * np.sin((angle -self.gamma)*np.pi/180)+grid_center_point_location[1]]



        return point_rotate_location

    def create_five_well_grid(self,range_x,range_y):
        '''
        创建五点式井网
        :param range_x: 横轴矩形范围
        :param range_y: 纵轴矩形范围
        :return:
        '''
        x_num=0
        y_num=0
        x_len=self.cell_len*self.x_zoom
        y_len=self.cell_len*self.y_zoom*np.sin(self.theta*np.pi/180)
        points_array={}
        while y_num*y_len*np.sin(self.theta*np.pi/180)+range_y[0]<=range_y[1]:
            points_array[y_num]=[]

            while x_len*x_num+ range_x[0]<=range_x[1]:

                if y_num ==0:

                    point_LT_location=[y_len*np.cos(self.theta*np.pi/180)+x_num*x_len+range_x[0],y_len*np.sin(self.theta*np.pi/180)+range_y[0]]
                    point_center_location=[ (x_num+0.5)*x_len+0.5*y_len*np.cos(self.theta*np.pi/180)+range_x[0],0.5*y_len*np.sin(self.theta*np.pi/180)+range_y[0] ]
                    points_array[y_num].append([y_num,x_num,point_LT_location,point_center_location])
                    x_num=x_num+1

                else:

                    later_start_x=points_array[y_num-1][0][2][0]
                    start_x_location=later_start_x-(np.floor((later_start_x-range_x[0])/x_len)+1)*x_len
                    point_LT_location = [start_x_location+y_len * np.cos(self.theta * np.pi / 180) + x_num * x_len,
                                         y_len * np.sin(self.theta * np.pi / 180)+y_num*y_len*np.sin(self.theta*np.pi/180)+range_y[0]]
                    point_center_location = [start_x_location+(x_num + 0.5) * x_len + 0.5 * y_len * np.cos(self.theta * np.pi / 180),
                                             0.5 * y_len * np.sin(self.theta * np.pi / 180)+y_num*y_len*np.sin(self.theta*np.pi/180)+range_y[0]]
                    points_array[y_num].append([y_num, x_num, point_LT_location, point_center_location])
                    x_num=x_num+1
            else:
                y_num=y_num+1
                x_num=0
        else:
            pass

        grid_center_point_location=[(range_x[1]+range_x[0])/2,(range_y[1]+range_y[0])/2]

        for points_row in points_array:
            points_line = points_array[points_row]
            for point in points_line:
                point_LT = point[2]
                point_LT_rotate=self.grid_rotate(grid_center_point_location, point_LT)
                point[2]=[point_LT_rotate[0]+self.Delta_x,point_LT_rotate[1]+self.Delta_y]

                point_center = point[3]
                point_center_rotate=self.grid_rotate(grid_center_point_location, point_center)
                point[3]=[point_center_rotate[0]+self.Delta_x, point_center_rotate[1]+self.Delta_y]

        return points_array

def draw_scatter(points_array,xlim,ylim):
    for points_row in points_array:
        points_line = points_array[points_row]
        for point in points_line:
            point_LT = point[2]
            point_center = point[3]
            plt.scatter(point_LT[0], point_LT[1], marker='o', color='red', s=40, label='First')
            plt.scatter(point_center[0], point_center[1], marker='x', color='red', s=40, label='First')
            plt.xlim(xlim[0],xlim[1])
            plt.ylim(ylim[0],ylim[1])
    plt.show()



if __name__=="__main__":

    grid=well_grid_class()

    grid.cell_len = 100
    grid.x_zoom = 1.2
    grid.y_zoom = 1.5
    grid.theta = 60
    grid.gamma = 30
    grid.Delta_x =20
    grid.Delta_y = 30

    range_x=[-500,1500]
    range_y=[-500,1500]
    points_array=grid.create_five_well_grid(range_x,range_y)

    xlim=[0,1000]
    ylim=[0,1000]
    draw_scatter(points_array,xlim,ylim)



