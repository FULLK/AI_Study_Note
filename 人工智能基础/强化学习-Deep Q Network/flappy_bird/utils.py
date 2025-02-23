import torch
import cv2
import numpy as np

#将图像转换为 PyTorch 张量。
def image_to_tensor(image):
    # 之前是(84, 84, 1)，下面一行就会把数据变成 (1, 84, 84)
    #对原始图像数组进行维度转置。在 OpenCV 中，图像的标准形状通常是 (height, width, channels)，
    # 而 PyTorch 更倾向于使用 (channels, height, width) 这种格式。
    image_tensor = image.transpose(2, 0, 1)
    #将图像的数据类型转换为 float32，这是因为 PyTorch 中的张量通常使用浮点型数据
    image_tensor = image_tensor.astype(np.float32)
    #将 NumPy 数组转换为 PyTorch 张量
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
    return image_tensor

#输入图像进行一些预处理，包括裁剪、调整大小、灰度转换和二值化
def resize_and_bgr2gray(image):
    # 把flappy bird每一帧图像里面的地面去除掉，小鸟每一时刻的向上或向下的选择和地面没有关系，让模型把注意力别放在地面上面
    #图像裁剪为 288x404
    image = image[:288, :404]  # 512*0.79=404
    # 改变大小以及变成黑白图像
    #将图像的大小调整为 84x84 像素 将彩色图像转换为灰度图
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    # 二值化 将灰度图中的所有非黑色像素（值大于 0）设置为 255，即将图像二值化。
    image_data[image_data > 0] = 255
    # reshape改变维度 将图像的维度调整为 (84, 84, 1)，即 84x84 的灰度图（只有一个颜色通道）
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data