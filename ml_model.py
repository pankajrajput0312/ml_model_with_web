import numpy as np
import os
import cv2
import joblib

model = joblib.load('iris.pkl')
classes = {"0": "Setosa", "1": "Versicolour", "2": "Virginica"}


def predict(sepal_length, sepal_width, petal_length, petal_width):
    input = [[sepal_length, sepal_width, petal_length, petal_width]]
    output = model.predict(input)
    class_name = classes[str(output[0])]
    return class_name


if __name__ == "__main__":
    print(predict(10, 11, 12, 13))
