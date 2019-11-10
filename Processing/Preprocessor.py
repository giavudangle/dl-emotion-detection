import cv2 as cv


class Preprocessor:
    # Constructor
    def __init__(self, width, height, interpolation=cv.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    # Resize image
    def preprocess(self, image):
        return cv.resize(image, (self.width, self.height), interpolation=self.interpolation)
