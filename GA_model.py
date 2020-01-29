import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import  mplot3d
from matplotlib.font_manager import FontProperties

import copy

class GA():
    def __init__(self):
        self.range=[2,4]
        self.num=100
        self.n_max=300
        self.cross_rate=0.6
        self.mutate_rate=0.03
        self.save_rate=0.3


    def Func(self,x,y):
        Func_value = x*(np.sin(x**2))+y*(np.sin(y**2))
        # Func_value=-( (x-3)**2+(y-3)**2 )
        return Func_value

    def save(self,pop_list):

        value_list=[]
        pop_list_GA=[]
        pop_list_save=[]

        for x_y in pop_list:
            x=x_y[0]
            y=x_y[1]
            value=self.Func(x,y)
            value_list.append(value)

        value_list=np.array(value_list)
        idx_list = np.argsort(value_list)[::-1]

        idx_save_list=idx_list[0:int(self.num*self.save_rate)]
        idx_GA_list=idx_list[int(self.num*self.save_rate)::]

        pop_list=np.array(pop_list)

        pop_list_save=pop_list[idx_save_list]

        pop_list_GA=pop_list[idx_GA_list]

        return pop_list_GA,pop_list_save


    def create_pop(self):

        pop_list=[]
        for i in range(self.num):
            x=np.random.randint( self.range[0], self.range[1])
            y=np.random.randint(self.range[0], self.range[1])
            location=[x,y]
            if location not in pop_list:
                pop_list.append(location)
            else:
                while location in pop_list:
                    x = np.random.rand()*(self.range[1]-self.range[0])+self.range[0]
                    y = np.random.rand()*(self.range[1]-self.range[0])+self.range[0]
                    location = [x, y]
                pop_list.append(location)
        return pop_list

    def select(self,pop_list):    # ���ݸ���ѡ��
        pop_len=len(pop_list)

        fitness_list=[]
        for x_y in pop_list:
            x=x_y[0]
            y=x_y[1]
            fitness=self.Func(x,y)
            fitness_list.append(fitness)
        min_value=np.min(fitness_list)

        for i in range(len(fitness_list)):
            fitness_list[i]=fitness_list[i]-min_value+0.000001

        fitness_list=np.array(fitness_list)

        idx = np.random.choice(np.arange(pop_len), size=pop_len, replace=True,p=(fitness_list/fitness_list.sum()))
        pop_list=np.array(pop_list)
        return pop_list[idx]

    def cross(self,pop_list,n_now,n_max):
        a=(n_max-n_now)/n_max

        pop_copy_list=copy.deepcopy(pop_list)

        child_list=[]

        for x_y in pop_list:

            if np.random.rand()<self.cross_rate:
                pop_copy_len=len(pop_copy_list)
                x=x_y[0]
                y=x_y[1]
                fitness = self.Func(x, y)

                idx=np.random.randint(0,pop_copy_len)
                x_y_copy=pop_copy_list[idx]
                x_copy=x_y_copy[0]
                y_copy=x_y_copy[1]
                # pop_copy_list = np.delete(pop_copy_list,idx,axis=0)

                x_child=a*x_copy+(1-a)*x
                y_child=a*y_copy+(1-a)*y
                fitness_child = self.Func(x_child, y_child)

                if fitness_child >=fitness:
                    child_list.append([x_child,y_child])
                else:
                    child_list.append([x, y])
            else:
                child_list.append(list(x_y))
        return child_list

    def mutate(self,child_list):  # ����
        pop_list_mutate = []
        for x_y in child_list:

            x=x_y[0]
            y=x_y[1]

            if np.random.rand()<self.mutate_rate:
                x=np.random.randint(self.range[0], self.range[1])

            if np.random.rand()<self.mutate_rate:
                y=np.random.randint(self.range[0], self.range[1])
            pop_list_mutate.append([x, y])
        pop_list_mutate=np.array(pop_list_mutate)
        return pop_list_mutate


if __name__ =="__main__":

    GA=GA()
    pop_list=GA.create_pop()

    mean_list=[]
    max_list=[]
    i_list=[]

    for i in range(GA.n_max):
        i_list.append(i)

        pop_list_GA,pop_list_save=GA.save(pop_list)

        pop_list_GA=GA.select(pop_list_GA)

        pop_list_GA=GA.cross( pop_list_GA, i, GA.n_max)

        pop_list_GA = GA.mutate(pop_list_GA)

        pop_list=np.concatenate((pop_list_save,pop_list_GA),axis=0)




        # ax = plt.axes(projection='3d')
        # ax.view_init(60, 35)
        # X = np.linspace(*GA.range, 20)
        # Y = np.linspace(*GA.range, 20)
        # X, Y = np.meshgrid(X, Y)
        # Z = GA.Func(X, Y)
        # # ax.plot_wireframe(X, Y, Z)
        # ax.plot_surface(X, Y, Z,alpha=0.7)

        x_list_show=[]
        y_list_show = []
        z_list_show = []


        for x_y in pop_list:
            x=x_y[0]
            y=x_y[1]
            z=GA.Func(x,y)

            x_list_show.append(x)
            y_list_show.append(y)
            z_list_show.append(z)

        mean_value=np.mean(z_list_show)
        max_value=np.max(z_list_show)
        mean_list.append(mean_value)
        max_list.append(max_value)

        print(mean_value)

    # plt.scatter(i_list, max_list, marker='o', color='red', s=2, label='First')
    # plt.show()

    plt.plot(i_list, mean_list, marker='x', mec='blue',  lw=1,ms=5,label='��Ⱥƽ��ֵ')
    plt.plot(i_list, max_list, marker='o', mec='red',  lw=1,ms=5,label='��Ⱥ����ֵ')
    font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc")
    plt.title('�Ľ��Ŵ��㷨���', fontproperties=font)
    plt.legend(prop=font)
    plt.show()


    #     if 'sca' in globals(): sca.remove()
    #     sca = ax.scatter3D(x_list_show, y_list_show, z_list_show, s=200, lw=0, c='red', alpha=0.5)
    #     plt.pause(0.05)
    # plt.ioff()
    # plt.show()


