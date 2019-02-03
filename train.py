# -*- coding: utf-8 -*-

"""
    文件名：train.py
    功能：主程序
    任务：搭建深度神经网络CNN，并进行图片分类
    数据集来源：https://www.cs.toronto.edu/~kriz/cifar.html
"""
from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import os


# 是否加载已经训练好的模型
load_model = False

# 模型保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

model_file = os.path.join(output_path, 'trained_cnn_model.h5')

# 图像大小
img_rows, img_cols = 32, 32

# 批大小
batch_size = 32

# 类别个数
n_classes = 10

# 迭代次数，一个epoch表示所有训练样本的一次前向传播和一次反向传播
epochs = 100

# 是否对样本进行增强
data_augmentation = True


def load_cifar10_dataset():
    """
        加载cifar-10数据集
    """
    # 加载数据
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    print('训练集形状', X_train.shape)
    print('训练集有{}个样本'.format(X_train.shape[0]))
    print('测试集有{}个样本'.format(X_test.shape[0]))

    # 将图片像素值转换为0-1
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # 将标签转换为one hot编码
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    return X_train, y_train, X_test, y_test


def build_cnn():
    """
        构建一个CNN框架
    """
    print('构建CNN')
    input_shape = (img_rows, img_cols, 3)
    model = Sequential()

    # 第一层
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # 第二层
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 全连接层
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    print(model.summary())

    return model


def main():
    """
        主函数
    """
    X_train, y_train, X_test, y_test = load_cifar10_dataset()

    # 是否读取已保存的模型
    if not load_model:
        print('训练模型...')
        # 构建CNN
        cnn_model = build_cnn()
        # 训练模型
        cnn_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
        # 保存模型
    else:
        print('加载训练好的模型...')
        if os.path.exists(model_file):
            cnn_model = load_model(model_file)
        else:
            print('{}模型文件不存在'.format(model_file))
            return

    test_loss, test_acc = cnn_model.evaluate(X_test, y_test)
    print('测试loss: {:.6f}，测试准确率：{:.2f}'.format(test_loss, test_acc))


if __name__ == '__main__':
    main()
