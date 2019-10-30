import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt


def get_ori_pop(num_points,num_pop,pop_max):

    ori_pop=[]
    n=0
    while n < num_pop:
        pop=[]
        i=0
        while i < num_points:#生成一套方案
            locationA = np.random.randint(0, pop_max)
            if locationA not in pop:
                pop.append(locationA)
                pop.sort()
                i = len(pop)
        if pop not in ori_pop:
            ori_pop.append(pop)
            n=len(ori_pop)

    ori_pop=np.array(ori_pop)

    return ori_pop

def crossover(parent_a, parent_b_list,cross_rate,pop_max):     # 随机交叉

    if np.random.rand() < cross_rate:

        index_parent_b = np.random.randint(0, len(parent_b_list), size=1)[0]
        parent_b=parent_b_list[index_parent_b]
        index_cross_list = np.random.randint(0, 2, size=len(parent_b)).astype(np.bool)
        child = []
        for i in range(len(parent_a)):
            is_cross=index_cross_list[i]
            if is_cross == True:
                if parent_b[i] not in child:
                    child.append(parent_b[i])
                elif parent_a[i] not in child :
                    child.append(parent_a[i])
                else:
                    is_add=0
                    while i ==0:
                        locationA = np.random.randint(0, pop_max)
                        if locationA not in child:
                            child.append(locationA)
                            is_add = 1
            else:
                if parent_a[i] not in child:
                    child.append(parent_a[i])
                else:
                    is_add=0
                    while i ==0:
                        locationA = np.random.randint(0, pop_max)
                        if locationA not in child:
                            child.append(locationA)
                            is_add = 1

        child.sort()

        child=np.array(child)

    else:
        child=parent_a

    return child

def mutate(child,MUTATION_RATE,pop_max): #变异

    for i in range(len(child)):
        if np.random.rand() < MUTATION_RATE:
            is_add=0
            while is_add ==0:
                location = np.random.randint(0, pop_max)
                if location not in child:
                    child[i] = location
                    is_add=1

    return child

def target_fun(data,points_list):
    target_value=0
    for points in points_list:
        value=data[points]
        target_value=target_value+value
    return target_value

def draw_fig(x_list,y_list,mat_info):

    plt.xlim(mat_info[2], mat_info[6])
    plt.ylim(mat_info[7], mat_info[3])
    plt.scatter(x_list,y_list)
    plt.pause(0.01)
    plt.cla()

def get_data_to_arr(path):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)  # DEM数据只有一种波段
    data = band.ReadAsArray()  # data即为dem图像像元的数值矩阵
    data = np.array(data)
    row,column=np.shape(data)
    data=data.flatten()
    geotransform = ds.GetGeoTransform()
    minx = geotransform[0]
    maxy = geotransform[3]
    step_length_Column = geotransform[1]
    step_length_Row=geotransform[5]
    maxx = minx + band.XSize * geotransform[1]
    miny = maxy + band.YSize * geotransform[5]
    mat_info=[row,column,minx,maxy,step_length_Column,step_length_Row,maxx,miny]
    return data,mat_info

def data_to_xy_list(arr,mat_info):
    x_list=[]
    y_list=[]
    for data in arr:
        row=int(np.floor((data+1)/mat_info[1]))
        column=(data+1)%mat_info[1]
        x=int(mat_info[2]+mat_info[4]*column)
        x_list.append(x)
        y=int(mat_info[3]+mat_info[5]*row)
        y_list.append(y)
    return x_list,y_list



if __name__=="__main__":


    '''
    GA模型参数
    '''
    GA_model={
        "num_points":50,
        "num_population":300,
        "num_iteration": 5000,
        "cross_rate":0.7,
        "mutate_rate":0.005
    }

    '''
    获取经编码后的栅格图像数组
    '''
    data,mat_info=get_data_to_arr('data_figure/埋深_100_西区.tif')
    '''
    生成初始种群
    '''
    pop_max=len(data)
    pop=get_ori_pop(GA_model['num_points'],GA_model['num_population'],pop_max)

    n_iteration_text=0
    max_result=0
    max_n_iteration=0
    pop_max_result = []

    for i in range(GA_model['num_iteration']):
        n_iteration_text = n_iteration_text + 1
        target_value_list = []
        for points_person in pop:
            target_value = target_fun(data, points_person)
            target_value_list.append(target_value)
        target_value_list = np.array(target_value_list)
        '''
        根据概率选择
        '''
        idx = np.random.choice(len(target_value_list), size=len(target_value_list), replace=True,
                               p=(target_value_list / target_value_list.sum()))
        pop=pop[idx]
        pop_copy=pop.copy()


        for parent in pop:
            '''
            交叉
            '''
            child=crossover(parent,pop_copy,GA_model['cross_rate'],pop_max)
            '''
            变异
            '''
            child = mutate(child,GA_model['mutate_rate'],pop_max)
            child.sort()
            parent = child



        i_text = 0
        mean_value = 0
        max_value = 0
        pop_max_value=[]
        for pop_list in pop:
            i_text = i_text + 1
            value = target_fun(data, pop_list)
            if value >max_value:
                max_value=value
                pop_max_value=pop_list

            mean_value=mean_value+value
        mean_value=mean_value/i_text

        if max_value > max_result:
            max_n_iteration=n_iteration_text
            max_result=max_value
            pop_max_result=pop_max_value

            '''
             转为x，y坐标，绘图
             '''
            x_list, y_list = data_to_xy_list(pop_max_result, mat_info)
            draw_fig(x_list, y_list, mat_info)

        if n_iteration_text %50==0:

            print('当前代数：',n_iteration_text,'均值：',int(mean_value),
                  '全局最大值所在代数：',max_n_iteration,'全局最大值：',int(max_result))
    print(pop_max_result)
















