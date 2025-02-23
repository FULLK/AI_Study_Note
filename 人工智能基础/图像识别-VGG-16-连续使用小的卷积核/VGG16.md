@[toc]

# VGG-16
这里的为啥一开始两个3*3卷积后大小还是224*224是因为存在padding使得卷积后大小不变
最后fc6和fc7都是4096个神经元的全连接层
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3e6544c839c54c648f9ad103492149d1.png)
AlexNet模型通过构造多层网络，取得了较好的效果，但是并没有给出深度神经网络设计的方向。VGG通过使用一系列大小为3x3的小尺寸卷积核和池化层构造深度卷积神经网络，并取得了较好的效果。VGG模型因为结构简单、应用性极强而广受研究者欢迎，尤其是它的网络结构设计方法，为构建深度神经网络提供了方向。

D是VGG-16

conv3-64代表卷积核是3*3 然后卷积核有64个通道
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a1e24f78f4d143c0803c21b838d226be.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c8b02076661a4b468bbce65a13c015a5.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2a1c202987254d3a9292be0d212be889.png)
这里的“简化”并不是指网络本身更简单，而是指网络的设计思想上避免了复杂的结构变化，通过将网络深度增加，采用统一的卷积核来提取更多层次的特征，最终获得更好的表现。
# 连续使用小的卷积核

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c17f161e34954a219e55989c8f832751.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c5fea904e9bf4f1b9b01feeed986b4e1.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2670119507724ce18aeace46715ad73b.png)

卷积后面一般会接上非线性变化，所有小卷积核会多次卷积，自然会有多次非线性变化
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3f6c9ba29b4f4f228f3b5ea9dfa67c61.png)

```bash
C:\Users\35925\anaconda3\envs\AI_Study\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\35925\anaconda3\envs\AI_Study\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
```

1. **`pretrained` 参数已弃用**：从 `torchvision` 版本 0.13 开始，`pretrained` 参数被弃用，应该使用 `weights` 参数来代替。库建议传入 `weights=VGG16_Weights.IMAGENET1K_V1` 或 `weights=VGG16_Weights.DEFAULT` 来加载预训练的权重。

2. **`weights` 参数的其他用法也被弃用**：除了 `pretrained` 外，传递给 `weights` 的其他参数也已经弃用，未来版本中可能会移除。推荐使用 `weights` 枚举类型来代替。

要解决这些警告，可以像下面这样更新代码：

```python
from torchvision.models import vgg16, VGG16_Weights

# 将弃用的 'pretrained' 参数替换为 'weights'
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
```

这样修改后，代码就符合了新版 API，并且可以消除警告。



# 实战
[数据集地址](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets)

# VGG16源码

##  vgg16
```python

@register_model()
@handle_legacy_interface(weights=("pretrained", VGG16_Weights.IMAGENET1K_V1))
def vgg16(*, weights: Optional[VGG16_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG16_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG16_Weights
        :members:
    """
    weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, weights, progress, **kwargs)

```

- VGG16_Weights.verify(weights) 验证 weights 参数的合法性。如果传入的值不是有效的 VGG16_Weights 枚举值，会报错。
- "D"：表示使用 VGG 的 D 配置（即 VGG-16）。
- False：表示不启用批量归一化（batch normalization）。
- weights：指定加载的权重。
## _vgg

```python
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model

```

- cfgs 是一个字典，定义了不同版本的 VGG 网络配置。
- 键（如 "A"、"B"、"D"、"E") 代表网络配置。
- 值是一个列表，描述网络中各层的结构：
- 数字（如 64, 128）：表示卷积层的输出通道数。
- "M"：表示池化层（MaxPooling）。

最后通过load_state_dict加载模型参数

## make_layers

```python
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

```

### 代码解析

`make_layers` 函数的主要作用是根据指定的配置 `cfg` 动态构建卷积层和池化层序列，并返回一个 `nn.Sequential` 容器。

---

### 参数解析

```python
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
```

1. **`cfg`**:  
   - 是一个列表，用于定义网络的结构。  
   - 列表中的元素可以是整数或字符串：
     - **整数**: 表示卷积层的输出通道数。
     - **字符串 `"M"`**: 表示池化层（MaxPooling）。  

2. **`batch_norm`**:  
   - 布尔值，决定是否在每个卷积层后添加批量归一化（Batch Normalization）。

3. 返回值:
   - 返回一个 `nn.Sequential` 对象，包含按 `cfg` 定义的所有层。

---

### 工作流程

1. 初始化空列表 `layers`，用于存储网络的各层模块：
   ```python
   layers: List[nn.Module] = []
   in_channels = 3  # 初始输入通道数为3（RGB图像）
   ```

2. 遍历配置列表 `cfg`，根据元素的类型（整数或 `"M"`）构建相应的层：
   ```python
   for v in cfg:
       if v == "M":
           layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 添加池化层
       else:
           v = cast(int, v)  # 将当前元素转换为整数（如果是整数，则不变）
           conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)  # 定义卷积层
           if batch_norm:
               layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]  # 带 BatchNorm
           else:
               layers += [conv2d, nn.ReLU(inplace=True)]  # 不带 BatchNorm
           in_channels = v  # 更新输入通道数
   ```

3. 将 `layers` 列表中的所有层包装成 `nn.Sequential` 模块，并返回：
   ```python
   return nn.Sequential(*layers)
   ```

---

### 每部分的功能

1. **池化层 (`"M"`)**:
   ```python
   layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
   ```
   - 当配置中出现 `"M"` 时，添加一个 `MaxPool2d` 层。
   - 该层使用 2×2 的窗口，并以步长 2 下采样，减小特征图的尺寸。

2. **卷积层**:
   ```python
   conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
   ```
   - 添加一个 3×3 的卷积层，并设置 `padding=1` 以保持特征图的尺寸不变。
   - `in_channels`: 输入特征图的通道数。
   - `v`: 卷积层的输出通道数（由配置定义）。

3. **批量归一化和激活函数**:
   - 如果 `batch_norm=True`，在卷积层后添加 `BatchNorm2d` 和 `ReLU` 激活函数：
     ```python
     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
     ```
   - 如果 `batch_norm=False`，只添加 `ReLU` 激活函数：
     ```python
     layers += [conv2d, nn.ReLU(inplace=True)]
     ```

4. **更新输入通道数**:
   ```python
   in_channels = v
   ```
   - 更新 `in_channels`，为下一层卷积层的输入通道数。

---

### 输出结果

函数返回一个 `nn.Sequential` 对象，包含由卷积层和池化层构成的网络。例如：

#### 示例 1: 配置为 `cfg = [64, "M", 128, "M"]`（无 BatchNorm）

```python
model = make_layers([64, "M", 128, "M"], batch_norm=False)
```

输出的网络结构为：
```
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU(inplace=True)
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```

#### 示例 2: 配置为 `cfg = [64, "M", 128, "M"]`（带 BatchNorm）

```python
model = make_layers([64, "M", 128, "M"], batch_norm=True)
```

输出的网络结构为：
```
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (6): ReLU(inplace=True)
  (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```

---


## VGG

```python

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

这段代码定义了一个 VGG 网络类（`VGG`），是基于 PyTorch 的神经网络模块 `nn.Module` 实现的。该类实现了 VGG 网络的基础结构，包括卷积特征提取部分、全连接分类部分以及初始化权重的方法。

---

### 1. **构造函数（`__init__`）**

#### 定义
```python
def __init__(
    self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
) -> None:
```
- **`features`**: 卷积特征提取部分的网络结构（通常通过 `make_layers` 函数生成）。
- **`num_classes`**: 分类任务的类别数量，默认为 1000（如 ImageNet 数据集的分类任务）。
- **`init_weights`**: 是否初始化权重，默认为 `True`。
- **`dropout`**: Dropout 的概率，默认为 0.5。

#### 初始化内容
```python
super().__init__()
_log_api_usage_once(self)
```
- 调用父类的初始化方法，确保继承了 `nn.Module` 的功能。
- `_log_api_usage_once(self)` 是用于记录模型 API 使用的日志（可能是为了统计功能）。

---

#### 模型组件

1. **卷积特征提取部分**:
   ```python
   self.features = features
   ```
   - `features` 是通过 `make_layers` 动态生成的卷积层序列，用于提取输入图像的特征。

2. **自适应平均池化**:
   ```python
   self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
   ```
   - 将特征图调整为固定大小的 `7×7`，无论输入图像尺寸是多少。

3. **全连接分类部分**:
   ```python
   self.classifier = nn.Sequential(
       nn.Linear(512 * 7 * 7, 4096),
       nn.ReLU(True),
       nn.Dropout(p=dropout),
       nn.Linear(4096, 4096),
       nn.ReLU(True),
       nn.Dropout(p=dropout),
       nn.Linear(4096, num_classes),
   )
   ```
   - `512 * 7 * 7` 是卷积层输出特征图展开后的大小。
   - 三层全连接：
     - 第一层：`512 * 7 * 7` → `4096`
     - 第二层：`4096` → `4096`
     - 第三层：`4096` → `num_classes`（类别数）。
   - 每层之间使用 `ReLU` 激活函数和 `Dropout` 正则化。

4. **初始化权重**:
   ```python
   if init_weights:
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)
   ```
   - 根据模块类型初始化权重：
     - **卷积层 (`nn.Conv2d`)**:
       - 使用 `kaiming_normal_` 初始化权重（适合 ReLU 激活函数）。
       - 将偏置初始化为 0。
     - **批量归一化层 (`nn.BatchNorm2d`)**:
       - 权重初始化为 1，偏置初始化为 0。
     - **全连接层 (`nn.Linear`)**:
       - 权重初始化为均值为 0，标准差为 0.01 的正态分布。
       - 偏置初始化为 0。

---

### 2. **前向传播（`forward`）**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)  # 卷积特征提取
    x = self.avgpool(x)   # 自适应平均池化
    x = torch.flatten(x, 1)  # 展平为 (batch_size, 512*7*7)
    x = self.classifier(x)   # 全连接分类
    return x
```

- 输入 `x` 经过以下步骤：
  1. **卷积特征提取部分**：
     - `self.features(x)` 提取卷积特征。
  2. **自适应平均池化**：
     - 将卷积特征图调整为固定大小 `7×7`。
  3. **展平操作**：
     - 使用 `torch.flatten` 将张量展平为二维形状：`(batch_size, 512*7*7)`。
  4. **全连接分类部分**：
     - 输入展平后的张量，经过 `self.classifier` 的全连接层进行分类。

---

