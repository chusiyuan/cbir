import processor
import numpy as np

if __name__ == "__main__":
    processor = processor.Processor('./dataset/image')
    # processor.preprocess()
    # processor.get_features_sift()
    # a = processor.cos_simi(np.array([1, 2, 3]), np.array([2, 2, 3]))
    # print(a)
    # gray = processor.load_im('./dataset/image/A0C573/A0C573_20151029151740_6565273745.jpg')
    # kps, fs = processor.get_feature_sift(gray)
    # print(type(fs))
    # print(fs.shape)
    # path = processor.get_im_path()
    # print(path)
    f = processor.bow()
    print(f)