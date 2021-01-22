# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import rasterio
import pandas as pd
from rasterio.transform import from_origin
from shapely.geometry import Point
import geopandas
import fiona

#=========================================================================================
# Делаю чистый слой для bias
raster = rasterio.open('layers/in/02.band1.asc')
array = np.array(raster.read()).ravel()
raster.profile

# {'driver': 'AAIGrid',
#  'dtype': 'int32',
#  'nodata': -9999.0,
#  'width': 5298,
#  'height': 1846,
#  'count': 1,
#  'crs': CRS({}),
#  'transform': (2594130.398597037,
#   2000.0,
#   0.0,
#   8276333.673568079,
#   0.0,
#   -2000.0),
#  'affine': Affine(2000.0, 0.0, 2594130.398597037,
#         0.0, -2000.0, 8276333.673568079),
#  'tiled': False}

df = pd.read_csv('M_nattereri.csv')

p = Point(0, 0)
x = p.buffer(1.0)

import shapely
ellipse=shapely.affinity.scale(x,15,20)

# geopandas.GeoDataFrame(df, geometry=ellipse)
geometry = [Point(xy).buffer(1.0) for xy in zip(df.iloc[:, 2], df.iloc[:, 1])]

res = geopandas.GeoDataFrame(df, geometry=geometry)

raster.shape

# (1846, 5298)
array.shape
# (9780108,)

result = []
for i in array:
    if i == raster.nodata:
        result.append(0)
    else:
        result.append(1)

len(result)

# 9780108


res = np.array(result).reshape(raster.shape)

raster.bounds
# BoundingBox(left=2594130.398597037, bottom=4584333.673568079, right=13190130.398597037, top=8276333.673568079)

raster.affine
# Охват 2594130.3985970369540155 : 8276333.6735680792480707
# Affine(2000.0, 0.0, 2594130.398597037,
#        0.0, -2000.0, 8276333.673568079)

raster.crs.data = {'proj' : 'moll', 'lon_0' : 30, 'x_0' : 3335846.22854, 'y_0' : -336410.83237, 'datum' : 'WGS84', 'units' : 'm'}

raster.crs
# CRS({'proj': 'moll', 'lon_0': 30, 'x_0': 3335846.22854, 'y_0': -336410.83237, 'datum': 'WGS84', 'units': 'm'})

from_origin(2594130.3985970369540155, 8276333.6735680792480707, 2000,-2000)
# Affine(2000.0, 0.0, 2594130.398597037,
#        0.0, 2000.0, 8276333.673568079)
with rasterio.open('layers/out/cleanRu.asc', 'w',
                   driver='AAIGrid',
                   width=raster.width, height=raster.height,
                   count=raster.count, dtype=raster.dtypes[0],
                   nodata=res[0][0],
                   transform=raster.affine) as dst:
    dst.write(res, 1)

#=========================================================================================
# Связь точкек в растре-таблице
# size of max: 2594130.3987940,4584333.6732685   left-bottom
#              2594130.3985970,4584333.6735680   rc2xy(1845.5, -0.5)
# size in min: 21.1695812304911,41.1789328201428 left-bottom

path = 'layers/out/cleanRu.asc'

from affine import Affine

from osgeo import gdal
ds = gdal.Open(path, gdal.GA_ReadOnly)
T0 = Affine.from_gdal(*ds.GetGeoTransform())
ds = None  # close

# T0 = layer.affine  так не работает после обновления пакетов 25.05.2019 использовать transform

T1 = T0 * Affine.translation(0.5, 0.5)
rc2xy = lambda r, c: (c, r) * T1

# https://stackoverflow.com/questions/27861197/how-to-i-get-the-coordinates-of-a-cell-in-a-geotif

print(rc2xy(1845.5, -0.5)) # row, column

# (2594130.398597037, 4584333.673568079)

print(rc2xy(1845, 0)) # посмотрел по пикселям в qgis, значение на 4 знаке меняется впределах клетки
# (2595130.398597037, 4585333.673568079)

T0 # для поиска ячейки по координате, нужно инвертировать эту матрицу
# Affine(2000.0, 0.0, 2594130.398597037,
#        0.0, -2000.0, 8276333.673568079)

~T0
# Affine(0.0005, 0.0, -1297.0651992985186,
#        0.0, -0.0005, 4138.166836784039)

# from numpy import matrix
# invT0 = matrix(T0).I
invT0 = ~T0

invT1 = invT0 * Affine.translation(0.5, 0.5)
xy2rc = lambda x, y: (x, y) * invT1

print(xy2rc(2595130.398597037, 4585333.673568079)) # Ура ура ура          column, row
# (0.5002500000000509, 1845.4997499999995)

#=================================================================---------------------------------------
# Всё успешно, нужно совместить имеющиеся точки по ночницам
cols = ['LONGITUDE', 'LATITUDE']
our_myotis = np.array(pd.read_csv('MNattereri_transformed_FIN.csv', usecols=cols)).tolist()
their_myotis = np.array(pd.read_csv('myotis.csv', usecols=cols)).tolist()

integer_our_myotis = []
integer_their_myotis = []
for i in our_myotis:
    integer_our_myotis.append([int(i[0]), int(i[1])])

for i in their_myotis:
    integer_their_myotis.append([int(i[0]), int(i[1])])

print(len(integer_our_myotis))
print(len(integer_their_myotis))

# 154
# 1531
integer_our_myotis[0]
# [4803592, 6166699]
for i in range(0, len(integer_our_myotis)):
    if integer_our_myotis[i] not in integer_their_myotis:
        their_myotis.append(our_myotis[i])
# np.append(integer_their_myotis, integer_our_myotis[0])

len(their_myotis)
# 1625

len(their_myotis) - 1531
# 94

pd.DataFrame(their_myotis, columns=cols).to_csv('Test_myotis_ru.csv')

len(integer_their_myotis) - 1531
# 92
# df_res = pd.DataFrame(integer_their_myotis, columns=cols)

df_res.to_csv('All_myotis_ru_test.csv')
their_myotis[0]
# array([4759717.12102252, 5892448.42980801])

print(integer_our_myotis[7] not in integer_their_myotis)
# True

for i in integer_their_myotis # ходи по 2 массивам и проверяй на equals


# =======================================--------------------------------------------------
# Сделал файл, получились точки, теперь нужно добавить буфер... Разрешение слоя 2км
raster = rasterio.open(path)
array = np.array(raster.read())[0]
df = pd.read_csv('All_myotis_ru.csv', usecols=cols)

points = np.array(df)
print(len(points))
# 1625

len(array[0])
# 5298

for p in points:
    row, col = xy2rc(p[0], p[1])
    array[int(col)][int(row)] = 10
    for i in range(1, 3):
        array[int(col)][int(row)-i] = 10 # по часовой стрелке
        array[int(col)+i][int(row)-i] = 10
        array[int(col)+i][int(row)] = 10
        array[int(col)+i][int(row)+i] = 10
        array[int(col)][int(row)+i] = 10
        array[int(col)-i][int(row)+i] = 10
        array[int(col)-i][int(row)] = 10
        array[int(col)-i][int(row)-i] = 10

    array[int(col)+1][int(row)-2] = 10
    array[int(col)+2][int(row)-1] = 10
    array[int(col)+2][int(row)+1] = 10
    array[int(col)+1][int(row)+2] = 10
    array[int(col)-1][int(row)+2] = 10
    array[int(col)-2][int(row)+1] = 10
    array[int(col)-2][int(row)-1] = 10
    array[int(col)-1][int(row)-2] = 10


with rasterio.open('Myotis_bias_FIN.asc', 'w',
                   driver=raster.driver,
                   width=raster.width, height=raster.height,
                   count=raster.count, dtype=raster.dtypes[0],
                   nodata=raster.nodata,
                   transform=T0) as dst:
    dst.write(array, 1)