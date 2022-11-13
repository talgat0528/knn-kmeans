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
    num = 10
    numOfClustering = len(clusteringList)
    for i in range(numOfClustering):
        errors = []
        for k in range(num):
            optimal_error = []
            for j in range(num):
                init_centers = cluster_init(clusteringList[i],k+1)
                _,error = kmeans(clusteringList[i],np.asarray(init_centers))
                optimal_error.append(error)
            errors.append(min(optimal_error))
            print(str(k+1) + '/' + str(num))
        print('finished ' + str(i+1) + '/' + str(numOfClustering))
        plt.plot(range(1,11), errors)
        plt.xlabel('k values')
        plt.ylabel('final objective value')
        plt.title('k vs final objective values on clustering ' + str(i+1))
        plt.show()