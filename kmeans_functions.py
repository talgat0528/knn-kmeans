import numpy as np
def calculate_distance(X1, X2):
    distance = (sum((X1 - X2)**2))**0.5
    return distance
def assign_clusters(data, cluster_centers):
    closest_center = []
    for i in data:
        dist = []
        for j in cluster_centers:
            dist.append(calculate_distance(i,j))
        closest_center.append(np.argmin(dist))
    return closest_center
    """
    Assigns every data point to its closest (in terms of euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """


def calculate_cluster_centers(data, assignments, cluster_centers, k):
    new_cluster_centers = cluster_centers.copy()
    for i in set(assignments):
        new_cluster_centers[i,:] = np.mean(data[assignments == i,:],axis = 0)
    
    return new_cluster_centers
    """
    Calculates cluster_centers such that their squared euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """


def kmeans(data, initial_cluster_centers):
    threshold = 1e-3
    cluster_centers = initial_cluster_centers.copy()
    old_cluster_centers = initial_cluster_centers.copy()
    obj_func = 0.0
    minimized_obj_func = 0.0
    while True:
        assignments = assign_clusters(data,cluster_centers)
        for i in set(assignments):
            relevant_data = data[assignments == i,:]
            for j in relevant_data:
                obj_func += sum((j-cluster_centers[i])**2)
        minimized_obj_func = obj_func
        obj_func = 0.0
        cluster_centers = calculate_cluster_centers(data,assignments,cluster_centers,len(cluster_centers))
        if np.linalg.norm(cluster_centers-old_cluster_centers)<threshold:
            break
        old_cluster_centers = cluster_centers
    #print(cluster_centers)
    return cluster_centers, minimized_obj_func
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
