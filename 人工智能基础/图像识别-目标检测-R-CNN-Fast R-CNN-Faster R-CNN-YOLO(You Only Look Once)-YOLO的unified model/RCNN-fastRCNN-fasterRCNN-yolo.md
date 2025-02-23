@[toc]

# 目标检测
找出图片中物体，并用矩形框框出来，并给出类别和对应概率
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/60382c5fd48c48d2a2a0dcdd35e9f2b9.png)

## 早期目标检测流程
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d463289716a5433b939e1d70a36e9124.png)

---

### **第一阶段：通过计算机图像学找出大量候选框**
- 在左侧的原始图像中，包含目标（小狗和小猫）以及背景。
- 系统首先使用传统的图像处理技术（如滑动窗口、选择性搜索、边缘检测等）生成**大量候选框**。
- 候选框是覆盖图像中潜在目标区域的矩形框，图中右侧的绿色矩形框表示这些候选框。
- 候选框的生成是目标检测的第一步，目的是减少需要分析的区域。

---

### **第二阶段：依次带入机器学习分类模型进行判断**
- 生成候选框后，逐一将每个候选框输入**分类器**（如支持向量机或简单的神经网络）。
- 分类器的任务是判断每个候选框内是否包含目标物体，以及目标的类别（如小狗或小猫）。
- 如果某个候选框通过分类器被判断为目标，则会被保留下来；否则将被丢弃。
- 右侧图中，保留下来的候选框可能覆盖小狗和小猫等目标。

---

### **总结**
这种方法的核心是**两步走**：
1. 先生成尽可能多的候选框，确保目标区域不会被遗漏。
2. 再利用分类模型逐一判断，确定目标的存在及其类别。

---

### **局限性**
这种早期方法的主要问题是：
- 候选框数量多，计算代价高（特别是对大图像）。
- 分类器需要针对每个候选框单独运行，导致效率低下。
- 候选框可能包含许多重叠区域，进一步增加计算复杂度。

现代目标检测算法（如YOLO、Faster R-CNN）通过引入**端到端检测**的思想，直接在整张图像中高效地预测目标和位置，解决了这些问题。

## 选择搜索(Selective Search)
 ![ ](https://i-blog.csdnimg.cn/direct/ddc238c11c424db29d94edab518f47df.png)
---

### **选择性搜索方法简介**
1. **背景介绍：**
   - 选择性搜索（Selective Search）是一种生成候选框的方法，主要用于目标检测。
   - 它由 **Koen E.A** 于2011年提出，是一种经典的候选区域生成算法。

2. **核心思想：**
   - 图像中的物体可能存在的区域往往是一些具有相似性（如颜色、纹理等）或连续性的区域。
   - 选择性搜索基于**区域合并**的思想，结合图像分割技术，将具有相似性特征的区域逐步合并，生成一系列候选框。

---

### **选择性搜索的主要流程**
1. **图像分割：**
   - 首先对输入图像进行**初步分割**，将图像划分为多个较小的子区域。
   - 图中**上方第一排的彩色分割图**展示了不同分割阶段的结果，每种颜色表示一个分割区域。

2. **区域合并：**
   - 将相邻的子区域根据相似性指标（例如颜色、纹理、大小等）不断合并，逐步形成更大的区域。
   - 每次合并后都会在这些区域上生成候选框（bounding boxes），即图中**下方第二排蓝色矩形框**。

3. **候选框生成：**
   - 最终输出的是一组包含潜在目标的矩形候选框，用于后续分类器判断目标是否存在。

---

### **选择性搜索的特点**
1. **优点：**
   - 不依赖滑动窗口方法，因此比传统的滑动窗口更高效且灵活。
   - 利用了图像本身的特性（颜色、纹理等）来生成候选区域，减少了计算成本。

2. **不足：**
   - 生成的候选框数量仍然较多，可能包含冗余信息（大量重叠框）。
   - 对于高分辨率图像，分割和区域合并过程计算较慢。

---

### **总结**
选择性搜索的核心是在图像中借助层次聚类的思想，通过将相似的区域逐步合并，生成候选目标区域（候选框）。  
其主要优点是结合了图像分割技术，比滑动窗口方法更高效；但在速度和框的数量上仍有改进空间，因此在现代目标检测方法（如Faster R-CNN）中逐渐被更高效的算法取代。

# 	R-CNN
### **R-CNN 简介**
R-CNN（Regions with Convolutional Neural Networks）是目标检测领域的一个里程碑方法，由 Ross Girshick 于 2014 年提出。R-CNN 的核心思想是将图像目标检测问题分解为两个步骤：
1. **候选区域生成**：通过传统算法（如 Selective Search）生成可能包含目标的候选框。
2. **目标分类**：使用卷积神经网络（CNN）对这些候选框逐一提取特征，并使用分类器（如 SVM）对其分类，同时用回归器调整候选框的位置。

尽管 R-CNN 的检测精度高，但其速度较慢，因为需要对每个候选框单独运行 CNN。

---

### **R-CNN 的主要步骤**

1. **输入图像：**
   - 输入原始图片。
   
2. **生成候选区域：**
   - 使用 Selective Search 等方法生成大约 2000 个候选框（Regions of Interest, RoI）。

3. **特征提取：**
   - 将每个候选框单独裁剪并调整大小（如 \(224 \times 224\)），然后通过 CNN 提取特征。

4. **分类与边界框调整：**
   - 对每个候选框提取的特征，使用分类器（如 SVM）判断目标类别。
   - 使用回归器对候选框的边界进行优化，生成更精确的目标框。

---


#  Fast R-CNN
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/95b16d5732874c6ea5621a5a2b48d6f6.png)
最后的终端将SVM替换为了Softmax分类器
采用了SPP-net的方式，将整张图像带入ConvNet中，之后从feature map中映射出proposal对应的位置
借鉴了SPP-net的方式，只不过RoI Pooling layer是SPP layer的特例
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/066fda773bba47d0acdd0d4c6d579647.png)
虽然依然是two stage framework
反向传播的时候，整体可以one stage training
但是请注意，提候选框的算法依然是Selective search，基于原图来提特征，并且不可训练
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d4b9bad3657048b28d734156a179bd20.png)
Multi-task loss: 使用一个多任务损失函数，包括分类损失（Log loss）和回归损失（Smooth L1 loss）。

- Log loss（分类损失）:

用于分类任务，衡量模型对目标类别预测的误差。
例如，如果候选框被标注为 "cat"，但模型预测为 "dog"，则会产生分类损失。
- Smooth L1 loss（边界框回归损失）:

用于回归任务，衡量预测边界框与真实边界框之间的误差。
使用 Smooth L1 函数代替普通的 L2 损失，以减少对异常值的敏感性。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ac770ab3fced40a2bab449bad5628711.png)



## SPP-net（Spatial Pyramid Pooling Network，空间金字塔池化网络）

SPP-net 是一个用于图像分类和目标检测的深度学习网络，由何凯明（Kaiming He）等人在 2014 年提出。它的核心是引入了 SPP 层（Spatial Pyramid Pooling Layer，空间金字塔池化层），从而解决了传统 CNN 模型对输入图片尺寸固定的要求。



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7173c00d2f9446b9ab8bc8b4bd37c96d.png)

# Faster R-CNN
最重要的就是引入了RPN网络，取代SS来提取候选框！！！（ss就是selective search ）

原来是最开始拿到原图去找候选框，现在是经过CNN得到特征图再去经过RPN拿到候选框（proposal ）

会有四个损失函数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/416c3e41aaae4fc4af4abc429a78dd19.png)
选择搜索办法速度慢，切换成SS后速度加快，是1/0.2=5FPS（每秒帧）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d7fa4017a1b34e478a2c3726a088b136.png)

# YOLO（You Only Look Once）
将目标检测任务看成是一个简单的回归问题去看待。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4c06445fc1ac49dbb776ff412e5c8ff5.png)
分类器是类别概率，回归器是候选框

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1e155b742ba043fe9ca231a3849eedbe.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c8c84372caa74d85a365735baa6f0051.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3bac12b044e545e682201852436587a1.png)

# YOLO的unified model
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/36b71ef1086e46bc82e83a266744ca6e.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3aee6b8823eb4f9bb43d03e35fe23306.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a78af357007b41e9a820849de8cacb25.png)
YOLO会将输入图像看成是 个网格组成的，例如，在上图中，person（蓝色边框）的中心落在了黄色grid内，因而黄色grid负责对person进行预测。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/53aa9e3d8b3f4d84b041a12f41f296cb.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/daec891853634c178938388c73ca6598.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea9e8a814c6c4bb398deb512787c2061.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/774a8d27df87451a8e851d7c014f63be.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ac1586cae5974274a3862de88cd8b1f0.png)
