import numpy as np
from collections import Counter
def calculate_distances(train_data, test_datum):
    distance = np.linalg.norm(train_data - test_datum,axis = 1)
    return distance
    """
    Calculates euclidean distances between test_datum and every train_data
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param test_datum: A (D, ) shaped numpy array
    :return: An (N, ) shaped numpy array that contains distances
    """


def majority_voting(distances, labels, k):
    nearest_neigbors = distances.argsort()[:k]
    expected = labels[nearest_neigbors]
    expected = sorted(expected)
    class_counter = Counter()
    for neighbor in expected:
        class_counter[neighbor]+=1
    return class_counter.most_common(1)[0][0]
    """
    Applies majority voting. If there are more then one major class, returns the smallest label.
    :param distances: An (N, ) shaped numpy array that contains distances
    :param labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: An integer. The label of the majority class.
    """


def knn(train_data, train_labels, test_data, test_labels, k):
    correct = 0
    for i in range(len(test_data)):
        dist = calculate_distances(train_data, test_data[i])
        dist = majority_voting(dist,train_labels,k)
        #correct+= (dist == test_labels[i])
        if(dist == test_labels[i]):
            correct+=1
    return float(correct/len(test_data))
    

        
    """
    Calculates accuracy of knn on test data using train_data.
    :param train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param train_labels: An (N, ) shaped numpy array that contains labels
    :param test_data: An (M, D) shaped numpy array where M is the number of examples
    and D is the dimension of the data
    :param test_labels: An (M, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :return: A float. The calculated accuracy.
    """


def split_train_and_validation(whole_train_data, whole_train_labels, validation_index, k_fold):
    train_data_copy = list(whole_train_data)
    train_labels_copy = list(whole_train_labels)
    fold_size = int(whole_train_data.shape[0]/k_fold)
    train_data = []
    train_labels = []
    validation_data = []
    validation_labels = []
    trainset = whole_train_data.shape[0]-fold_size
    for i in range(k_fold):
        if(validation_index == i):
            for j in range(fold_size):
                validation_data.append(train_data_copy[(i*fold_size)+j])
                validation_labels.append(train_labels_copy[(i*fold_size)+j])
        else:
            for j in range(fold_size):
                train_data.append(train_data_copy[(i*fold_size)+j])
                train_labels.append(train_labels_copy[(i*fold_size)+j])
            
    result = np.asarray(train_data), np.asarray(train_labels), np.asarray(validation_data), np.asarray(validation_labels)
    return result
    """
    Splits training dataset into k and returns the validation_indexth one as the
    validation set and others as the training set. You can assume k_fold divides N.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param validation_index: An integer. 0 <= validation_index < k_fold. Specifies which fold
    will be assigned as validation set.
    :param k_fold: The number of groups that the whole_train_data will be divided into.
    :return: train_data, train_labels, validation_data, validation_labels
    train_data.shape is (N-N/k_fold, D).
    train_labels.shape is (N-N/k_fold, ).
    validation_data.shape is (N/k_fold, D).
    validation_labels.shape is (N/k_fold, ).
    """


def cross_validation(whole_train_data, whole_train_labels, k, k_fold):
    avgacc = 0.0
    totalacc = 0.0
    for i in range(k_fold):
        train_data, train_labels, validation_data, validation_labels = split_train_and_validation(whole_train_data, whole_train_labels,i, k_fold)
        totalacc += knn(train_data,train_labels,validation_data,validation_labels,k)
    avgacc = totalacc/k_fold
    return avgacc
    
    """
    Applies k_fold cross-validation and averages the calculated accuracies.
    :param whole_train_data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param whole_train_labels: An (N, ) shaped numpy array that contains labels
    :param k: An integer. The number of nearest neighbor to be selected.
    :param k_fold: An integer.
    :return: A float. Average accuracy calculated.
    """
