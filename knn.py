import numpy as np
import matplotlib.pyplot as plt
from knn_functions import calculate_distances, majority_voting,knn,split_train_and_validation, cross_validation
k = list(range(1,200))
k_fold = 10

train_data = np.load('knn/train_data.npy')
train_labels = np.load('knn/train_labels.npy')
test_data = np.load('knn/test_data.npy')
test_labels = np.load('knn/test_labels.npy')
y = []
best = (0.0,0.0)
for i in range(len(k)):
    res = cross_validation(train_data,train_labels,k[i],k_fold)
    y.append(res)
    if(res>best[1]):
        best = (k[i],res)

accuracy_on_test = knn(train_data,train_labels,test_data,test_labels,best[0])
print(accuracy_on_test)
plt.plot(k,y)
plt.xlabel('k values')
plt.ylabel('average accuracy')
plt.title('average accuracies from the cross-validation')
plt.show()

