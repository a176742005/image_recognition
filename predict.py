# -*- coding: utf-8 -*-

"""
    文件名：predict.py
    功能：实现图像的分类
    任务：构建输入图像预处理操作，并对图像进行预测
"""

import cv2
from keras.models import load_model
import numpy as np
import os


def image_recognition(path):
    """
          函数功能：当输入一张图片时，输出其所对应的类型
          函数参数：字符串
    """
    # 读取所需要识别的图片数据
    image = cv2.imread('C:/picture/{}'.format(path))
    # 将图片的大小转变为 32*32*3 的大小，满足训练集的图像维度。
    image = cv2.resize(image, (32, 32))
    # opencv读取数据是 B，G，R 格式，我们需要将 B，G，R 格式转化为 R，G，B 格式。
    B, G, R = cv2.split(image)
    image = cv2.merge([R, G, B])
    # 对需要识别的数据进行归一化操作
    image = image.astype('float32')
    image /= 255
    # 导入keras模型
    cnn_model = load_model(r"F:\cnn\cnnn\output\trained_cnn_model.h5")
    # 利用训练好的模型对图像进行预测，因为训练集数据为四维，所以将图像也转化为四维。
    value = cnn_model.predict(np.array([image]), batch_size=50)
    # 输出value数组中最大值所对应的下标
    max_index = np.argmax(value)
    # 对照字典
    check_dict = {
          0: '飞机',
          1: '汽车',
          2: '鸟',
          3: '猫',
          4: '鹿',
          5: '狗',
          6: '青蛙',
          7: '马',
          8: '船',
          9: '货车'
    }
    # 输出识别结果
    print(check_dict[max_index])


def main():
    """
          主函数
    """
    name = os.listdir("C:\\picture\\")
    for i in name:
        print(i+" 的识别结果为：", end='')
        image_recognition(i)


if __name__ == '__main__':
    main()