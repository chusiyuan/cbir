import cv2
import numpy as np
import os
import pickle


class GetFeatures:
    def __init__(self, data, k):
        self.test_im_path = '.\\dataset\\image\\A3XE95\\A3XE95_20151203073732_6787841316.jpg'
        self.file_type = ['jpg', 'bmp', 'png']
        self.K = k
        self.features_num = []
        if type(data) == str:
            self.path_list = self.get_im_path(data)
        elif type(data) == list:
            self.path_list = data

        # with open('path_list_2.pk', 'wb') as f:
        #     pickle.dump(self.path_list, f)
        self.all_features = self.get_all_features()
        self.features_bow, self.centers = self.bow()
        self.features_tf_idf, self.idf = self.tf_idf()
        # np.save('all_features_2.npy', self.all_features)
        # np.save('features_bow_2.npy', self.features_bow)
        # np.save('features_centers_2.npy', self.centers)
        # np.save('features_tf_idf_2.npy', self.features_tf_idf)
        # np.save('features_idf_2.npy', self.idf)


    def load_im(self, im_path):
        im = cv2.imread(im_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return im_gray

    def get_feature_sift(self, im):
        sift = cv2.xfeatures2d.SIFT_create()
        # surf = cv2.xfeatures2d.SURF_create()
        kps, features = sift.detectAndCompute(im, None)
        return kps, features

    def get_im_path(self, path):
        path_list = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                for ftype in self.file_type:
                    if ftype in filename:
                        path_list.append(os.path.join(dirpath, filename))
                        break
        path_list.remove(self.test_im_path)
        return path_list

    def get_all_features(self):
        all_features = np.zeros((2, 128))
        for i in range(len(self.path_list)):
            im_gray = self.load_im(self.path_list[i])
            kps, features = self.get_feature_sift(im_gray)
            self.features_num.append(features.shape[0])
            all_features = np.vstack((all_features, features))
            # print('got im' + str(i) + '\'s features')
        all_features = np.float32(all_features)
        all_features = all_features[2:]
        return all_features

    def bow(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(self.all_features, self.K, None, criteria, 1, flags)
        index = 0
        features_after_bow = np.zeros((len(self.features_num), self.K))
        for i in range(len(self.features_num)):
            labels_i = labels[index: index + self.features_num[i]]
            index = index + self.features_num[i]
            for j in range(self.K):
                features_after_bow[i][j] = sum(sum(labels_i == j))
        # np.save('features_after_bow.npy', features_after_bow)
        return features_after_bow, centers

    def tf_idf(self):
        tf = np.zeros(shape=self.features_bow.shape)
        idf = np.zeros(shape=self.features_bow[0].shape)
        for i in range(self.features_bow.shape[0]):
            sum_i = sum(self.features_bow[i])
            for j in range(self.features_bow.shape[1]):
                tf[i][j] = self.features_bow[i][j]/sum_i
                idf[j] = idf[j] + (self.features_bow[i][j] != 0)
        idf = self.features_bow.shape[0] / (idf + 1)
        features_tf_idf = tf * idf
        return features_tf_idf, idf

