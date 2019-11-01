import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import  mplot3d
#王彦迪
#2018 11 20

DNA_SIZE = 16        # 基因长度
POP_SIZE = 50         # 群体数量
CROSS_RATE = 0.9      # 交叉概率
MUTATION_RATE = 0.01 # 变异概率
N_GENERATIONS = 101   # 迭代次数
X_BOUND = [2,4]    # X范围（设Y相同划分，不另设参数）
pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # 初始化种群矩阵


def Func(x,y):
    # if (y<0):
    #     Func_value=-((x+5)**2+(y-4)**2)+10
    # else:
    #     Func_value=-((x-5)**2+(y-4)**2)+10
    Func_value = x*(np.sin(x**2))+y*(np.sin(y**2))
    return Func_value#目标函数

def translateDNA(pop):#解码 将二进制转换为十进制的实际值，pop1为x，pop2为y
    pop1=((pop[:,0:int(DNA_SIZE/2)].dot(2 ** np.arange(DNA_SIZE / 2)[::-1]) / float(2 ** (DNA_SIZE / 2) - 1)) * (X_BOUND[1]-X_BOUND[0]))+X_BOUND[0]
    pop2=((pop[:,int(DNA_SIZE/2):int(DNA_SIZE)].dot(2 ** np.arange(DNA_SIZE / 2)[::-1]) / float(2 ** (DNA_SIZE / 2) - 1)) * (X_BOUND[1]-X_BOUND[0]))+X_BOUND[0]
    return pop1,pop2

def select(pop, fitness):    # 根据概率选择



    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness/fitness.sum()))
    return pop[idx]


# def crossover(parent, pop):     # 随机交叉
#     if np.random.rand() < CROSS_RATE:
#         i_ = np.random.randint(0, POP_SIZE, size=1)
#
#         # cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)
#         cross_pointsx = np.random.randint(0, 2, size=int(DNA_SIZE/2)).astype(np.bool)
#         cross_pointsy = np.random.randint(0, 2, size=int(DNA_SIZE/2)).astype(np.bool)
#
#
#         parentx = parent[0:int(DNA_SIZE/2)]
#         parenty = parent[int(DNA_SIZE/2):int(DNA_SIZE)]
#
#
#         # parent[cross_points] = pop[i_, cross_points]
#
#         popx = pop[:, 0:int(DNA_SIZE/2)]
#         popy = pop[:, int(DNA_SIZE/2):int(DNA_SIZE)]
#
#         parentx[cross_pointsx] = popx[i_, cross_pointsx]
#         parenty[cross_pointsy] = popy[i_, cross_pointsy]
#
#         parent = np.concatenate((parentx, parenty), axis=0)
#     return parent

def crossover(parent, pop):     # 单点交叉
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)

        # cross_pointsx = np.random.randint(0, 2, size=12).astype(np.bool)
        # cross_pointsy = np.random.randint(0, 2, size=12).astype(np.bool)

        parentx = parent[0:int(DNA_SIZE/2)]
        parenty = parent[int(DNA_SIZE/2):int(DNA_SIZE)]

        # parent[cross_points] = pop[i_, cross_points]
        popx = pop[i_, 0:int(DNA_SIZE/2)]
        popy = pop[i_, int(DNA_SIZE/2):int(DNA_SIZE)]
        popx = popx.flatten()
        popy = popy.flatten()

        cross_pointx = np.random.randint(0, int(DNA_SIZE/2))
        cross_pointy = np.random.randint(0, int(DNA_SIZE/2))

        parentx = np.concatenate((parentx[0:cross_pointx], popx[cross_pointx:int(DNA_SIZE/2)]), axis=0)
        parenty = np.concatenate((parenty[0:cross_pointy], popx[cross_pointy:int(DNA_SIZE/2)]), axis=0)


        parent = np.concatenate((parentx, parenty), axis=0)
    return parent

def mutate(child): #变异
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


def get_fitness(pred):
    return pred+ 1e-3  - np.min(pred) #由于在选择过程中，和作为分母，因此取正



Allresult=np.zeros((N_GENERATIONS,4))

def get_bestresult(Allresult,i,x,y,fitness):
    Allresult[i,:]=np.concatenate(([i],x, y,fitness), axis=0)
    return Allresult

for i in range(N_GENERATIONS):

    Func_values=Func(translateDNA(pop)[0],translateDNA(pop)[1])

    ax = plt.axes(projection='3d')
    ax.view_init(60,35)
    X = np.linspace(*X_BOUND, 20)
    Y = np.linspace(*X_BOUND, 20)
    X, Y = np.meshgrid(X, Y)
    Z = Func(X, Y)
    ax.plot_wireframe(X, Y, Z)
    #ax.plot_surface(X, Y, Z,alpha=0.7)

    if 'sca' in globals(): sca.remove()
    sca = ax.scatter3D(translateDNA(pop)[0],translateDNA(pop)[1],Func_values,s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)
    fitness = get_fitness(Func_values)
    result=np.array([pop[np.argmax(fitness), :]])
    Allpoints=get_bestresult(Allresult,i, translateDNA(result)[0].flatten(), translateDNA(result)[1].flatten(),Func(translateDNA(result)[0], translateDNA(result)[1]).flatten())

    if i%10==0:
        print("fitted DNA: ", i,translateDNA(result)[0],translateDNA(result)[1],Func(translateDNA(result)[0],translateDNA(result)[1]))

    pop = select(pop, fitness)

    pop_copy = pop.copy()

    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child

best_point=np.array([Allpoints[np.argmax(Allpoints[:,3]), :]])
print('最优结果为第 {} 代，X： {}  Y： {}  结果： {}'.format(int(best_point[0,0]),best_point[0,1],best_point[0,2],best_point[0,3]))

plt.ioff(); plt.show()


