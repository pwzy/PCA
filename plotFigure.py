import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

feature = np.load("./feature.npy") 

#  print(feature) [180, 2]
print(feature.shape)

normal_x, normal_y, normal_z = [], [], []
abnormal_x, abnormal_y, abnormal_z = [], [], []
for i in range(60):
    #  normal_x[i] = feature[i][0]
    #  normal_y[i] = feature[i][1]
    normal_x.append(feature[i][0])
    normal_y.append(feature[i][1])
    normal_z.append(feature[i][2])
for i in range(60, 180):
    #  abnormal_x[i] = feature[i][0]
    #  abnormal_y[i] = feature[i][1]
    abnormal_x.append(feature[i][0])
    abnormal_y.append(feature[i][1])
    abnormal_z.append(feature[i][2])

plt.scatter(normal_x, normal_y, c='r', marker='x')
plt.scatter(abnormal_x, abnormal_y, c='b', marker='D')
#  fig = plt.figure()
#  ax = Axes3D(fig)
#  ax.scatter3D(normal_x, normal_y, normal_z, c='b')
#  ax.scatter3D(abnormal_x, abnormal_y, abnormal_z, c='r')

plt.show()




