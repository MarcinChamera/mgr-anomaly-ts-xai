from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import imblearn
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def get_context(outlier, nearest_neighbors, inliers, input_features):
    neighbors_index = nearest_neighbors.kneighbors([outlier], return_distance=False).tolist()[0]
    return inliers.iloc[neighbors_index][input_features].values

def find_best_L_silhouette(C_i, max_L):
    silhouette_scores = []
    min_L = 2
    for i in range(min_L, max_L + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(C_i)
        silhouette_scores.append(silhouette_score(C_i, kmeans.labels_))
    best_silhouette_score = silhouette_scores.index(max(silhouette_scores)) + min_L
    return best_silhouette_score

def cluster(C_i, O_i, L):
    kmeans = KMeans(n_clusters=L, random_state=0).fit(C_i)
    C_i_l = np.array([C_i[kmeans.labels_ == i] for i in range(L)])
    O_i_l = np.array([O_i[kmeans.labels_ == i] for i in range(L)])
    return C_i_l, O_i_l

def upsample(outlier, inliers):
    ROS = imblearn.over_sampling.RandomOverSampler(sampling_strategy=1, random_state=0)
    X = np.vstack((outlier, inliers))
    y = np.hstack((
        np.ones((1, 1)),
        np.zeros((1, inliers.shape[0]))
    )).T
    X_resampled, Y_resampled = ROS.fit_resample(X, y)
    O_i_indices = [i for i in range(len(Y_resampled)) if Y_resampled[i] == 1]
    return X_resampled[O_i_indices]

def create_local_classifier(C_i_l, O_i_l, clf):
    if clf == 'svm':
        clf = svm.LinearSVC(penalty='l1', dual=False)
    elif clf == 'lr':
        clf = LogisticRegression(penalty='l1', solver='saga')
    X = np.vstack((C_i_l, O_i_l))
    y = np.hstack((np.zeros(C_i_l.shape[0]), np.ones(O_i_l.shape[0]))) 
    clf.fit(X, y)
    return clf

def abandon_small_clusters(clustered, not_clustered, ratio): # ratio=0.03 in the paper
    return np.array([cluster_i for cluster_i in clustered if len(cluster_i) >= ratio * len(not_clustered)])

def calculate_all_abnormal_attribute_scores(context_results, input_features_length):
    abnormal_attribute_scores = np.zeros(input_features_length)
    for context_cluster_results in context_results:
        abnormal_attribute_scores += len(context_cluster_results['context_cluster']) * \
            np.abs(context_cluster_results['clf'].coef_[0])
    abnormal_attribute_scores /= np.sum([len(context_result['context_cluster']) for context_result in context_results])
    abnormal_attribute_scores /= np.sum(abnormal_attribute_scores)
    return abnormal_attribute_scores

def calculate_outlierness_score(context_results, outlier):
    outlierness_score = 0
    for context_cluster_results in context_results:
        # decision_function will return probability instead on int label
        outlierness_score_cluster = abs(context_cluster_results['clf'].decision_function(outlier.reshape(1, -1))) /\
            np.linalg.norm(context_cluster_results['clf'].coef_[0], ord=2)
        outlierness_score += np.linalg.norm(len(context_cluster_results['context_cluster']) * outlierness_score_cluster * \
            context_cluster_results['clf'].coef_[0] / np.linalg.norm(context_cluster_results['clf'].coef_[0]))
    outlierness_score /= sum([len(context_result['context_cluster']) for context_result in \
                                 context_results])
    return outlierness_score

def coin(inliers, outliers, input_features, context_neighbors=50, abandon_ratio=0.05, max_clusters=5, predefined_L=None,
         local_clf='svm'):
    nearest_neighbors = NearestNeighbors(n_neighbors=context_neighbors, metric='euclidean')
    nearest_neighbors.fit(inliers[input_features].values)
    outlierness_score_list = []
    abnormal_attributes_scores_list = []
    for o_i in outliers[input_features].values:
        # get context C_i
        C_i = get_context(o_i, nearest_neighbors, inliers, input_features)
        # do upsampling of o_i to O_i so that len(O_i) = len(C_i)
        O_i = upsample(o_i, C_i)
        # find optimal number of clusters L to later divide C_i
        if predefined_L is not None:
            L = predefined_L
        else: 
            # silhouette score instead of prediction strength
            L = find_best_L_silhouette(C_i, max_clusters) 
            L = min(L + 1, max_clusters)
        # divide C_i into sufficiently exclusive clusters C_i_l for l = 1, ..., L
        # and divide O_i into the same number of clusters
        C_i_clusters, O_i_clusters = cluster(C_i, O_i, L)
        # clusters of small size are abandoned in subsequent procedures
        C_i_clusters = abandon_small_clusters(C_i_clusters, C_i, abandon_ratio)
        O_i_clusters = abandon_small_clusters(O_i_clusters, O_i, abandon_ratio)
        # for each pair of clusters C_i_l and O_i_l, create a local, simple, interpretable classifier
        context_results = []
        for C_i_l, O_i_l in zip(C_i_clusters, O_i_clusters):
            clf = create_local_classifier(C_i_l, O_i_l, local_clf)
            context_results.append({
                'clf': clf, 
                'context_cluster': C_i_l
                })
        # calculate abnormal attributes scores
        abnormal_attributes_scores = calculate_all_abnormal_attribute_scores(context_results, len(input_features))
        abnormal_attributes_scores_list.append(abnormal_attributes_scores)
        # outlierness score
        outlierness_score = calculate_outlierness_score(context_results, o_i) # odpowiednik devt_i u ninghao
        outlierness_score_list.append(outlierness_score)
    return outlierness_score_list, abnormal_attributes_scores_list