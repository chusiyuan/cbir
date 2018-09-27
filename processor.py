import cv2
import numpy as np
import math
import os

class Processor:
    def __init__(self, path):
        self.dataset_path = path
        self.file_type = ['jpg', 'bmp', 'png']
        self.path_list = []
        self.feature_list = []
        self.all_features = np.zeros((2, 128))
        self.features_num = []
        self.K = 100

    def load_im(self, im_path):
        im = cv2.imread(im_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return im_gray

    def euclidean(self, mat1, mat2):
        mat1 = mat1.reshape(1, -1)
        mat2 = mat2.reshape(1, -1)
        dif = mat1 - mat2
        euc = math.sqrt(np.dot(dif, dif.T)[0][0])
        return euc

    def cos_simi(self, mat1, mat2):
        mat1 = mat1.reshape(1, -1)
        mat2 = mat2.reshape(1, -1)
        cos = (np.dot(mat1, mat2.T)[0][0])/(math.sqrt(np.dot(mat1, mat1.T)[0][0])*math.sqrt(np.dot(mat2, mat2.T)[0][0]))
        return cos

    def get_feature_sift(self, im):
        sift = cv2.xfeatures2d.SIFT_create()
        # surf = cv2.xfeatures2d.SURF_create()
        kps, features = sift.detectAndCompute(im, None)
        return kps, features

    def get_im_path(self):
        # path_list = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                for type in self.file_type:
                    if type in filename:
                        self.path_list.append(os.path.join(dirpath, filename))
                        break
        # print(self.path_list)
        # return path_list

    def get_all_features(self):
        self.get_im_path()
        # all_features = np.zeros((2, 128))
        for i in range(10):
        #for i in range(len(self.path_list)):
            im_gray = self.load_im(self.path_list[i])
            kps, features = self.get_feature_sift(im_gray)
            # print(features.shape[0])
            self.features_num.append(features.shape[0])
            self.feature_list.append(features)
            self.all_features = np.vstack((self.all_features, features))
            print('got im' + str(i) + '\'s features')
        self.all_features = np.float32(self.all_features)
        self.all_features = self.all_features[2:]
        print(self.all_features.shape)
        print(self.features_num)

    def bow(self):
        self.get_all_features()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(self.all_features, self.K, None, criteria, 10, flags)
        print(labels.shape)
        print(type(labels))
        print(labels == 100)
        print(sum(labels == 10))
        index = 0
        features_after_bow = np.zeros((len(self.features_num), self.K))
        for i in range(len(self.features_num)):
            labels_i = labels[index: index + self.features_num[i]]
            index = index + self.features_num[i]
            for j in range(self.K):
                features_after_bow[i][j] = sum(sum(labels_i == j))
        return features_after_bow









    # def preprocess(self):
    #     im = cv2.imread(self.impath)
    #     self.im = im
    #     cv2.imshow('im', im)
    #     cv2.waitKey()
    #     print(im.shape)
    #     print(type(im))
    #     im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #     print(im_gray.shape)
    #     self.imgray = im_gray
    #     cv2.imshow('im2', self.imgray)
    #     cv2.waitKey()
    #
    #
    # def get_features_sift(self):
    #     sift = cv2.xfeatures2d.SIFT_create()
    #     featurepoints = sift.detect(self.imgray)
    #     print(len(featurepoints))
    #     print(featurepoints[0])
    #     cv2.imshow('im2', self.imgray)
    #     cv2.waitKey()
    #     im = cv2.drawKeypoints(self.imgray, featurepoints, None)
    #     cv2.imshow('im', im)
    #     cv2.waitKey()