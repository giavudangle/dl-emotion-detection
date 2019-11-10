import cv2 as cv
import os
import numpy as np


class DatasetLoader:

    # Constructor
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        # Vì preprocessors ở đây là 1 list nếu chuyền vào preprocessors là 1 None thì phải khởi tạo nó là 1 list
        if self.preprocessors is None:
            self.preprocessors = []

    # Load a list of image and preprocessing
    # image_paths là danh sách các đường dẫn ảnh, verbose là số ảnh được lưu vào bộ nhớ trong 1 lượt
    def load(self, image_paths, verbose=0, depth=3):
        # data lưu các hình ảnh, labels lưu nhãn của các hình ảnh tương ứng
        data, labels = [], []
        # enumerate giúp trả về vị trí i tương ứng. Tương tự for(int i=0; i<a.Len; i++)
        for i, image_path in enumerate(image_paths):
            if depth == 3:
                # Đọc ảnh màu
                image = cv.imread(image_path, cv.IMREAD_COLOR)
            else:
                # Đọc ảnh xám
                image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            # Lấy nhãn ảnh sau khi cắt bỏ kí tự '/', nằm tại vị trí thứ 2 từ phải sang trái
            label = image_path.split(os.path.sep)[-2]

            # Resize hình ảnh theo các kích thước cho trước trong list preprocessors nếu có
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # add vào data và labels
            data.append(image)
            labels.append(label)

            # Hiển thị thông tin mỗi lần load theo số lượng verbose
            if verbose > 0 and i > 0 and ((i + 1) % verbose == 0 or (i + 1) == len(image_paths)):
                print('[INFO]: Images processed: {}/{}'.format(i + 1, len(image_paths)))

        return np.array(data), np.array(labels)
