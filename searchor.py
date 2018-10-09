import numpy as np
import math
import heapq
import os


class Searchor:
    def __init__(self, features, path_list):
        self.features = features
        self.path_list = path_list
        self.file_type = ['jpg', 'bmp', 'png']

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

    def search(self, im_path, n):
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
        query_index = self.path_list.index(im_path)
        query = self.features[query_index]
        for i in range(len(self.features)):
            if i == query_index:
                simi_list.append(1)
            else:
                simi = self.cos_simi(query, self.features[i])
                simi_list.append(simi)
        # print(simi_list)
        result_index = list(map(simi_list.index, heapq.nlargest(n, simi_list)))
        # print(result_index)
        print('n = ' + str(n))
        # for i in result_index[1:]:
        for i in range(len(result_index)):
            name_i = self.path_list[result_index[i]].split('\\')[-2]
            if name_i == type:
                num += 1
            print(self.path_list[result_index[i]])
        print('------------------------------------------------------------')
        precision = num / n
        recall = num / type_num
        return precision, recall






