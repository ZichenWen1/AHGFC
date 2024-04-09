import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.io

color_map = ['r', 'y', 'k', 'g', 'b']  # 7个类，准备7种颜色


def plot_embedding_2D(data, cluster_labels, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[cluster_labels[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def plt_tsne(cluster_labels, data):
    # mat_data = scipy.io.loadmat('./data/wiki_cooc.mat')  # 根据你的.mat文件路径修改
    #
    # data = mat_data['node_features']

    # print("Features shape:", data.shape)
    #
    # print('Beginning clustering......')

    # 使用 KMeans 进行聚类
    # num_clusters = 5  # 你可以根据需要设置聚类的数量
    # kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    # cluster_labels = kmeans.fit_predict(data)

    # print('Finished clustering......')

    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    data = data.cpu().numpy()
    result_2D = tsne_2D.fit_transform(data)

    print('Finished t-SNE......')
    fig1 = plot_embedding_2D(result_2D, cluster_labels, 't-SNE with Clustering')
    plt.show()


