import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from osgeo import gdal
import ospybook as pb
import pandas as pd
import numpy as np

def read_tif(path):

    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)#DEM数据只有一种波段
    # ov_band=band.GetOverview(band.GetOverviewCount()-3)
    # data=ov_band.ReadAsArray()
    data = band.ReadAsArray()#data即为dem图像像元的数值矩阵
    return data,ds,band

def read_csv(path):
    data=np.array(pd.read_csv(path,header=None))
    return data

def data_csv_show_2D(path):
    data=read_csv('data/IDW_兰氏压力.csv')
    plt.imshow(data)
    plt.colorbar(shrink=.83)







if __name__ =="__main__":

    # data, ds, band=read_tif('data/gas2_Extract.tif')
    #
    # #计算边界坐标
    # geotransform = ds.GetGeoTransform()
    # minx = geotransform[0]
    # maxy = geotransform[3]
    # maxx = minx + band.XSize * geotransform[1]
    # miny = maxy + band.YSize * geotransform[5]
    # x = np.arange(minx, maxx, geotransform[1])
    # y = np.arange(maxy, miny, geotransform[5])
    # x, y = np.meshgrid(x[:band.XSize], y[:band.YSize])

    data=read_csv('data/IDW_兰氏压力.csv')

    '''
    三维显示
    '''
    # x=np.arange(0,data.shape[0])
    # y=np.arange(0,data.shape[1])
    # x, y = np.meshgrid(x, y)
    #
    # fig = plt.figure()
    #
    #
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(x, y, data, cmap='gist_earth', lw=0)




    plt.show()

