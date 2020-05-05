import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image

import msop

def bipartite(features_1, features_2):
    feature_xs_1, feature_ys_1, discriptors_1 = features_1
    feature_xs_2, feature_ys_2, discriptors_2 = features_2
    edges = []
    for i in range(len(feature_xs_1)):
        discriptor_1 = discriptors_1[i]
        discriptor_2 = discriptors_2[i]
        if discriptor_1.shape[0] > 1 and discriptor_2.shape[0] > 1:
            adj = np.zeros([discriptor_1.shape[0], discriptor_2.shape[0]])
            distance = np.zeros([discriptor_1.shape[0], discriptor_2.shape[0]])
            for a in range(discriptor_1.shape[0]):
                distance[a] = np.sum((discriptor_1[a] - discriptor_2) ** 2, (1, 2)) ** 0.5
            #fig, ax = plt.subplots()
            #ax.hist(distance[0], bins=10)
            #plt.show()
            #print(distance)
            k = 4
            top12 = np.argsort(distance, 1)[:, :k]
            top21 = np.argsort(distance, 0)[:k, :]
            '''
            for a in range(discriptor_1.shape[0]):
                fig, axes = plt.subplots(ncols=k+1)
                axes[0].imshow(discriptor_1[a])
                for kk in range(k):
                    axes[kk+1].imshow(discriptor_2[top12[a, kk]])
                plt.show()
            '''
            edge12 = np.zeros([discriptor_1.shape[0], discriptor_2.shape[0]], dtype=np.bool)
            edge21 = np.zeros([discriptor_1.shape[0], discriptor_2.shape[0]], dtype=np.bool)
            #print(top12)
            #print(top21)
            half = top12.shape[1] // 2
            for a in range(discriptor_1.shape[0]):
                #print(distance[a, top12[a, half:]])
                outlier_distance = np.mean(distance[a, top12[a, half:]])
                for j in range(half):
                    if distance[a, top12[a, j]] < 0.65 * outlier_distance:
                        edge12[a, top12[a, j]] = True
            half = top21.shape[0] // 2
            for b in range(discriptor_2.shape[0]):
                #print(distance[top21[half:, b], b])
                outlier_distance = np.mean(distance[top21[half:, b], b])
                for j in range(half):
                    if distance[top21[j, b], b] < 0.65 * outlier_distance:
                        edge21[top21[j, b], b] = True
            edge = edge12 & edge21
            print(np.sum(edge))
            edges.append(edge)
            #print('12', distance[0, top12[0]])
            #print('21', distance[top21[:, 0], 0])
        else:
            edges.append(None)
    return edges
def get_model(features_1, features_2):
    edges = bipartite(features_1, features_2)
    adj_list1 = []
    adj_list2 = []
    eg_list = []
    now1 = 0
    now2 = 0
    for edge in edges:
        n1 = edge.shape[0]
        n2 = edge.shape[1]
        adj_list1 += [[] for i in range(n1)]
        adj_list2 += [[] for j in range(n2)]
        for i in range(n1):
            for j in range(n2):
                if edge[i][j]:
                    adj1[now1+i].append(now2+j)
                    adj2[now2+j].append(now1+i)
                    eg_list.append((now1+i, now2+j))
        now1 += n1
        now2 += n2
    
def cylindrical_projection(arr, f):
    pass
if __name__ == '__main__':
    #name1 = 'parrington/prtn02.jpg'
    #name2 = 'parrington/prtn01.jpg'
    name1 = 'pack2_rot/IMG_0022.JPG'
    name2 = 'pack2_rot/IMG_0023.JPG'
    if 1 == 1:
        feat1 = msop.msop(name1, 1)
        feat2 = msop.msop(name2, 1)

        import pickle
        f = open('tmp.pck', 'wb')
        pickle.dump((feat1, feat2), f)
    else:
        import pickle
        f = open('tmp.pck', 'rb')
        feat1, feat2 = pickle.load(f)
    edges = bipartite(feat1, feat2)
    im1 = Image.open(name1)
    arr1 = np.asarray(im1)
    im2 = Image.open(name2)
    arr2 = np.asarray(im2)
    
    from matplotlib.patches import ConnectionPatch
    for i in range(len(feat1[0])):
        xs1 = feat1[0][i]
        ys1 = feat1[1][i]
        xs2 = feat2[0][i]
        ys2 = feat2[1][i]
        fig, axes = plt.subplots(ncols=2)
        axes[0].imshow(arr1)
        axes[1].imshow(arr2)
        axes[0].scatter(ys1, xs1)
        axes[1].scatter(ys2, xs2)
        for a in range(xs1.shape[0]):
            for b in range(xs2.shape[0]):
                if edges[i][a, b]:
                    xy1 = (ys1[a], xs1[a])
                    xy2 = (ys2[b], xs2[b])
                    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data", axesA=axes[0], axesB=axes[1])
                    #axes[0].add_artist(con)
                    axes[1].add_artist(con)
        plt.show()
