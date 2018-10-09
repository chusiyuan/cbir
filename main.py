import processor
import searchor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    processor = processor.Processor('.\\dataset\\image')
    # processor.preprocess()
    # processor.get_features_sift()
    # gray = processor.load_im('./dataset/image/A0C573/A0C573_20151029151740_6565273745.jpg')
    # kps, fs = processor.get_feature_sift(gray)
    # path = processor.get_im_path()
    # print(path)
    # features_after_bow = processor.bow()
    # print(features_after_bow)
    # np.save('features_after_bow.npy', features_after_bow)
    # features_after_bow = np.load('features_after_bow.npy')
    # features_tf_idf = processor.tf_idf(features_after_bow)
    # print(features_tf_idf)
    # np.save('features_tf_idf', features_tf_idf)

    processor.get_im_path()
    features = np.load('features_tf_idf.npy')

    searchor = searchor.Searchor(features, processor.path_list)
    precision_list = []
    recall_list = []
    for n in range(1, 31):
        precision, recall = searchor.search('.\\dataset\\image\\A1F201\\A1F201_20151030112626_6570241117.jpg', n)
        # precision, recall = searchor.search('.\\dataset\\image\\A0C573\\A0C573_20151029151740_6565273745.jpg', n)
        precision_list.append(precision)
        recall_list.append(recall)
    print(precision_list)
    print(recall_list)

    plt.plot(recall_list, precision_list)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()