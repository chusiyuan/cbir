import numpy as np
import math
import heapq
import os
import cv2


class Searchor:
    def __init__(self, features, path_list, centers, idf):
        self.features = features
        self.path_list = path_list
        self.file_type = ['jpg', 'bmp', 'png']
        self.centers = centers
        self.idf = idf

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

    def do_query(self, im_path):
        type_num = 0
        type = im_path.split('\\')[-2]
        im_dir = '\\'.join(im_path.split('\\')[:-1])
        # print(im_dir)
        # print(type)
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                if filename.split('.')[-1] in self.file_type:
                    type_num += 1
                    # print(type_num)
        query = np.zeros(shape=self.features[0].shape)
        im = cv2.imread(im_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kps, features = sift.detectAndCompute(im_gray, None)
        for i in range(features.shape[0]):
            max_simi = 0
            max_index = 0
            for j in range(self.centers.shape[0]):
                simi = self.cos_simi(features[i], self.centers[j])
                if simi > max_simi:
                    max_simi = simi
                    max_index = j
            query[max_index] += 1

        tf = np.zeros(shape=query.shape)
        s = sum(query)
        for i in range(query.shape[0]):
            tf[i] = query[i] / s

        query_tf_idf = tf * self.idf
        return query_tf_idf, type_num, type

    def search(self, query_tf_idf, n, type_num, type):
        num = 0
        simi_list = []
        for i in range(len(self.features)):
            simi = self.cos_simi(query_tf_idf, self.features[i])
            simi_list.append(simi)
        result_index = list(map(simi_list.index, heapq.nlargest(n, simi_list)))
        # print('n = ' + str(n))
        result = []
        for i in range(len(result_index)):
            name_i = self.path_list[result_index[i]].split('\\')[-2]
            if name_i == type:
                num += 1
            # print(self.path_list[result_index[i]])
            result.append(self.path_list[result_index[i]])
        # print('------------------------------------------------------------')
        precision = num / n
        recall = num / type_num
        return precision, recall, result, result_index

    def resort(self, features_tf_idf, centers, im_path, idf, result):
        query = np.zeros(shape=features_tf_idf[0].shape)
        im = cv2.imread(im_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kps, features = sift.detectAndCompute(im_gray, None)
        for i in range(features.shape[0]):
            max_simi = 0
            max_index = 0
            for j in range(centers.shape[0]):
                simi = self.cos_simi(features[i], centers[j])
                if simi > max_simi:
                    max_simi = simi
                    max_index = j
            query[max_index] += 1

        tf = np.zeros(shape=query.shape)
        s = sum(query)
        for i in range(query.shape[0]):
            tf[i] = query[i] / s

        query_tf_idf = tf * idf
        query_list = query_tf_idf.tolist()
        query_type = query_list.index(max(query_list))
        # print('query type:')
        # print(query_type)
        new_result = []
        temp_list = []
        for i in range(features_tf_idf.shape[0]):
            temp = features_tf_idf[i].tolist()
            temp_type = temp.index(max(temp))
            # print('temp type:')
            # print(temp_type)
            if query_type == temp_type:
                new_result.append(result[i])
            else:
                temp_list.append(result[i])
        new_result += temp_list
        return new_result

    def search2(self, im_path, new_query, n):
        num = 0
        type_num = 0
        simi_list = []
        type = im_path.split('\\')[-2]
        im_dir = '\\'.join(im_path.split('\\')[:-1])
        # print(im_dir)
        # print(type)
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                if filename.split('.')[-1] in self.file_type:
                    type_num += 1
        # print(type_num)
        query_tf_idf = new_query

        for i in range(len(self.features)):
            simi = self.cos_simi(query_tf_idf, self.features[i])
            simi_list.append(simi)
        result_index = list(map(simi_list.index, heapq.nlargest(n, simi_list)))
        # print('n = ' + str(n))
        result = []
        for i in range(len(result_index)):
            name_i = self.path_list[result_index[i]].split('\\')[-2]
            if name_i == type:
                num += 1
            # print(self.path_list[result_index[i]])
            result.append(self.path_list[result_index[i]])
        # print('------------------------------------------------------------')
        precision = num / n
        recall = num / type_num
        return precision, recall, result, result_index













