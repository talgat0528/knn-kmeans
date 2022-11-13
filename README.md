# K - Nearest Neighbor (KNN)
- The dataset is created by modifying and introducing some noise to the Iris Dataset.
- Since it is supervised learning the dataset is labeled and has separate train and test sets. It is in the form of saved numpy array and it is loaded from knn directory.
- Used k-fold cross-validation on training data to find a suitable k for the KNN algorithm and test my KNN algorithm using that k on test dataset. Specifically, for k of KNN = 1, 2, 3, 4, ..., 199 , the code applies k-fold cross-validation using training data only and plots the avarage accuracies from the cross-validation. The plot has kKNN values for its x-axis and average accuracies for each of them for its y-axis. Selected kCV=10 for the k-fold cross-validation.  
- To run the program make sure you have numpy and matplotlib on your computer, then type the following command in your terminal:  
$python3 knn.py
# K - Means Clustering
- The dataset resides in kmeans directory.
- The result of the k-means algorithm depends on the initial clusters. Therefore, I did more than one initialization for the same setting and took the best result among them to have more consistent results. For each k=1, 2, 3, ..., 10, restarted k-means many times (for example 10 times) and took the minimum of the final values of the objective function. The code plots the values where x axis denotes the values of k and y axis denotes the best final values of the objective function on that k value.
- To run the program make sure you have numpy and matplotlib on your computer, then type the following command in your terminal:  
$python3 kmeans.py  
- Applied k-means to each clustering data with their obtained k values.
- executing **$python3 clustering.py** will draw the cluster centers with different shapes so that the result can be easily understood.
