import numpy as np
import matplotlib.pyplot as plt
from kmeans_functions import calculate_distance,assign_clusters,calculate_cluster_centers,kmeans
def cluster_init(data,k):
    cluster_center_min = data.min()
    cluster_center_max = data.max()
    cluster_centers = []
    for centroid in range(k):
        centroid = np.random.uniform(cluster_center_min,cluster_center_max,data.shape[1])
        cluster_centers.append(centroid)
    return cluster_centers
if __name__ == "__main__":
    clusteringList = []
    clusteringList.append(np.load('kmeans/clustering1.npy'))
    clusteringList.append(np.load('kmeans/clustering2.npy'))
    clusteringList.append(np.load('kmeans/clustering3.npy'))
    clusteringList.append(np.load('kmeans/clustering4.npy'))
    colors = ['red','blue','green','black', 'purple','brown']
    for c in range(len(clusteringList)):
        init_centers = cluster_init(clusteringList[c],c+2)
        centroids,error = kmeans(clusteringList[c],np.asarray(init_centers)) 
        plt.figure()
        assignments = assign_clusters(clusteringList[c],centroids)
        for i in set(assignments):
            plt.scatter(clusteringList[c][assignments ==i,:][:,0],clusteringList[c][assignments ==i,:][:,1],c = colors[i],alpha = 0.2) 
        plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='yellow', s = 300,marker = '+')
        plt.show()