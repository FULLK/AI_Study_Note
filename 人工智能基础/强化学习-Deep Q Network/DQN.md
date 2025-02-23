@[toc]
#  Deep Q Network
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/58657a6bc20041ea8bc2b499adbcd338.png)
这种方式很适合格子游戏。因为格子游戏中的每一个格子就是一个状态，这是离散的，但在现实生活中，很多状态并不是离散而是连续的。所以我们可以通过神经网络来完成离散状态的任务，初始输入是状态，输出的各个行为对应的Q值（也可以理解为概率），这样初始输入就可以是连续的即状态可以是连续的

 
Q learning和DQN并没有根本的区别。只是DQN用神经网络，也就是一个函数（神经网络）替代了原来Q table而已。Deep network + Q learning = DQN
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cddf766fe5ba4669a96e6634fd3064b2.png)

这里的Q值就是神经网络最后输出的不同行为的概率
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca9213f377944c0fa30f48f81fd6af7e.png)

#  zip(*batch)的内部实现

首先，来看下 `batch` 的构造：

```python
batch = sample(history_memory, min(len(history_memory), args.batch_size))
```

`batch` 是通过从 `history_memory` 中采样得到的，假设 `history_memory` 是一个包含多个经验的列表，其中每个经验是一个四元组（`state, reward, next_state, terminal`）。每个元素看起来像这样：

```python
[state, reward, next_state, terminal]
```

当你通过 `zip(*batch)` 时，`*batch` 会将 `batch` 中的每个四元组解包成四个不同的列表或元组，分别对应 `state`, `reward`, `next_state` 和 `terminal`。

## 假设：
```python
batch = [
    (state_1, reward_1, next_state_1, terminal_1),
    (state_2, reward_2, next_state_2, terminal_2),
    (state_3, reward_3, next_state_3, terminal_3)
]
```

当你执行 `zip(*batch)` 时，`*batch` 会解包成如下的四个参数传递给 `zip()`：

```python
zip((state_1, reward_1, next_state_1, terminal_1), 
    (state_2, reward_2, next_state_2, terminal_2), 
    (state_3, reward_3, next_state_3, terminal_3))
```

然后，`zip()` 函数会按位置将这些元组的元素“对齐”在一起，结果如下：

```python
[(state_1, state_2, state_3), 
 (reward_1, reward_2, reward_3), 
 (next_state_1, next_state_2, next_state_3), 
 (terminal_1, terminal_2, terminal_3)]
```

## 结果：
`zip(*batch)` 会返回一个迭代器，生成的是一个包含四个元组的列表（假设 `batch` 中有三个经验）：

```python
(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
```

这四个元组分别对应：

- `state_batch`: 包含所有的 `state` 值（`state_1, state_2, state_3`）
- `reward_batch`: 包含所有的 `reward` 值（`reward_1, reward_2, reward_3`）
- `next_state_batch`: 包含所有的 `next_state` 值（`next_state_1, next_state_2, next_state_3`）
- `terminal_batch`: 包含所有的 `terminal` 值（`terminal_1, terminal_2, terminal_3`）

