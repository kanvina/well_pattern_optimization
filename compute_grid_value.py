import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from create_well_grid import well_grid_class


def get_cell_value(data_array,x_range,y_range,cell_len,location):
    row=int( np.floor((y_range[1]-location[1])/cell_len))
    column=int(np.floor( (location[0]-x_range[0])/cell_len ))
    point_value=data_array[row,column]
    return point_value

def draw_scatter(points_array,xlim,ylim):
    for point in points_array:
        point_location=point[1]
        if point[0]==0:
            plt.scatter(point_location[0], point_location[1], marker='o', color='red', s=40, label='First')
        else:
            plt.scatter(point_location[0], point_location[1], marker='x', color='red', s=40, label='First')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

def create_well_grid_func(grid_range_x,grid_range_y,grid_info):
    grid=well_grid_class()
    grid.cell_len = grid_info['cell_len']
    grid.x_zoom = grid_info['x_zoom']
    grid.y_zoom = grid_info['y_zoom']
    grid.theta = grid_info['theta']
    grid.gamma = grid_info['gamma']
    grid.Delta_x =grid_info['Delta_x']
    grid.Delta_y = grid_info['Delta_y']
    points_array=grid.create_five_well_grid(grid_range_x,grid_range_y)

    well_points_array=[]
    for points_row in points_array:
        points_line = points_array[points_row]
        for point in points_line:
            point_LT = point[2]
            well_points_array.append([0,point_LT])
            point_center = point[3]
            well_points_array.append([1,point_center])
    return well_points_array

def get_grid_value(well_points_array,data_array,x_range,y_range,cell_len):
    # points_value=[]
    num=0

    sum_value=0
    for points in well_points_array:
        point_location=points[1]
        try:
            point_value=get_cell_value(data_array,x_range,y_range,cell_len,point_location)
            sum_value=sum_value+point_value
            num=num+1
            # if points[0]==0:
            #     points_value.append([0,point_location])
            # else:
            #     points_value.append([1, point_location])
        except:
            continue

    # return sum_num_mean_list, points_value
    return sum_value,num

def compute_sum_mean(data_array,x_range,y_range,cell_len,grid_info):

    grid_len=1.4*max(x_range[1]-x_range[0],y_range[1]-y_range[0])

    center_location=[(x_range[1]+x_range[0])/2,(y_range[1]+y_range[0])/2]
    grid_range_x=[center_location[0]-grid_len/2,center_location[0]+grid_len/2]
    grid_range_y=[center_location[1]- grid_len/2,center_location[1]+ grid_len/2]

    well_points_array=create_well_grid_func(grid_range_x, grid_range_y,grid_info)
    sum_value,well_num=get_grid_value(well_points_array,data_array,x_range,y_range,cell_len)
    # draw_scatter(points_value, x_range, y_range)
    return sum_value,well_num

if __name__ =="__main__":

    data_array=np.array( pd.read_csv('data/data_target.csv',header=None))

    x_range=[615887,635437]
    y_range=[3946571,3960751]
    data_cell_len=100


    grid_info={

        'cell_len':2000,
        'x_zoom':1,
        'y_zoom':1,
        'theta':60,
        'gamma':15,
        'Delta_x':0,
        'Delta_y':0
    }

    sum_num_mean_list=compute_sum_mean(data_array,x_range,y_range,data_cell_len,grid_info)














