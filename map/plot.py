import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle

def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin

with open('outData.p','rb') as fp:
		formatted_data, results, centroids = pickle.load(fp)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

color = ('r', 'b', 'g', 'c', 'm')
marker = ('.', '^', ',', 'o', 'v')
clusterSet = set(results)
clusterSet = list(clusterSet)

clusterMap = {}
for i in range(len(clusterSet)):
	clusterMap[str(clusterSet[i])] = (color[i], marker[i])	

for i in range(len(formatted_data)):
    xs = formatted_data[i][0]
    ys = formatted_data[i][1]
    zs = results[i]
    color = clusterMap[str(results[i])][0]
    m = clusterMap[str(results[i])][1]

    ax.scatter(xs, ys, zs, c=color, marker=m)

print("Finished plotting")

ax.set_xlabel('Latitutde')
ax.set_ylabel('Longitude')
ax.set_zlabel('Rating')

plt.show()