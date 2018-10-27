import searchor
import numpy as np
import matplotlib.pyplot as plt
import pickle
import get_features


def pr_draw(im_path):
    features = np.load('features_tf_idf_1.npy')
    with open('path_list.pk', 'rb') as f:
        path_list = pickle.load(f)
    centers = np.load('features_centers_1.npy')
    idf = np.load('features_idf_1.npy')
    s = searchor.Searchor(features, path_list, centers, idf)
    precision_list = []
    recall_list = []
    query, type_num, type = s.do_query(im_path)
    for n in range(1, 31, 1):
        # print(n)
        precision, recall, result, result_index = s.search(query, n, type_num, type)
        # precision, recall, result = s.search('.\\dataset\\image\\A1F201\\A1F201_20151030112626_6570241117.jpg', n)
        # precision, recall = searchor.search('.\\dataset\\image\\A0C573\\A0C573_20151029151740_6565273745.jpg', n)
        precision_list.append(precision)
        recall_list.append(recall)
    # print(precision_list)
    # print(recall_list)

    plt.plot(recall_list, precision_list)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim((0, 1.1))
    plt.ylim((0, 1.1))
    plt.show()


def test(im_path):
    features = np.load('features_tf_idf_2.npy')
    with open('path_list_2.pk', 'rb') as f:
        path_list = pickle.load(f)
    centers = np.load('features_centers_2.npy')
    idf = np.load('features_idf_2.npy')
    s = searchor.Searchor(features, path_list, centers, idf)
    # precision, recall, result, result_index = s.search('.\\dataset\\image\\A1F201\\A1F201_20151030112626_6570241117.jpg', 30)
    # '.\\dataset\\image\\A1NS85\\A1NS85_20151031103101_3022940881.jpg'
    query, type_num, type = s.do_query(im_path)
    precision, recall, result, result_index = s.search(query, 30, type_num, type)
    g = get_features.GetFeatures(result, 15)
    print(result)
    new_result = s.resort(g.features_tf_idf, g.centers, im_path, g.idf, result)
    print(new_result)

    precision_list_1 = []
    precision_list_2 = []
    for n in range(1, 26, 1):
        precision, recall, result, result_index = s.search(query, n, type_num, type)
        precision_list_1.append(precision)
        new_query = np.zeros(shape=features[0].shape)
        for i in range(len(result_index)):
            new_query += features[result_index[i]]
        new_query /= len(result_index)
        precision2, recall2, result2, result_index2 = s.search2(im_path, new_query, n)
        precision_list_2.append(precision2)

    plt.plot(range(1, 26, 1), precision_list_1)
    plt.plot(range(1, 26, 1), precision_list_2)
    plt.show()


if __name__ == "__main__":
    # g = get_features.GetFeatures('.\\dataset\\image', 100)
    # pr_draw('.\\dataset\\image\\A1F201\\A1F201_20151030112626_6570241117.jpg')
    test('.\\dataset\\image\\A3XE95\\A3XE95_20151203073732_6787841316.jpg')
    # test('.\\dataset\\image\\A1C133\\A1C133_20151129074004_6762833645.jpg')

