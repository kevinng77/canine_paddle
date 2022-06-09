# 训练要素总结

google 官方对微调训练的配置集中在 [run_tydi_lib.py](https://github.com/google-research/language/blob/b76d2230156abec5c8d241073cdccbb36f66d1de/language/canine/tydiqa/run_tydi_lib.py#L190) 和 [tydi_modeling.py](https://github.com/google-research/language/blob/master/language/canine/tydiqa/tydi_modeling.py)。前者为训练配置，如超参、io路径等。后者为模型训练架构的设计，如loss的选择、优化器等。

## 目录
  - [训练要素总结](#训练要素总结)
  - [1. 模型架构](#1-模型架构)
  - [2. 超参](#2-超参)
  - [3. 训练模式](#3-训练模式)
    - [3.1 优化器](#31-优化器)
    - [3.2 loss 计算](#32-loss-计算)
  - [4. 数据处理](#4-数据处理)
  - [5. 任务预测](#5-任务预测)
  - [6. 模型笔记](#6-模型笔记)
    - [6.1 大致架构](#61-大致架构)
      - [6.1.1 Embedding](#611-embedding)
      - [6.1.2 down sampling](#612-down-sampling)
      - [6.1.3 Deep transformer stack](#613-deep-transformer-stack)
      - [6.1.4 Upsampling](#614-upsampling)

## 1. 模型架构

TydiQA 任务使用的是 CanineQA 模型，需要在 Canine 基础上添加三个不同的 `dense` 来分别预测答案开始位置、答案结束位置、答案类型。参考官方代码：`tydi_modeling.py` [link](https://github.com/google-research/language/blob/master/language/canine/tydiqa/tydi_modeling.py)。

本次复现使用 `CanineTydiQA` 模型与官方设置一致：初始化 weight 采用 0.02 截尾正太分布，bias采用 0。

## 2. 超参

由于官方论文中并没有提到微调的参数配置，因此本次复现参考并分别尝试了 [canine官方仓库](https://github.com/google-research/language/tree/master/language/canine/tydiqa#train-fine-tuning-canine) 的微调配置（`batch_size=512`，`epoch=10`, `lr=5e-5`），以及 [tydiqa 基线仓库]() 的微调配置（`batch_size=16`,`epoch=3`, `lr=5e-5`）。其中 `batch_size=512` 通过梯度累加来近似模拟。除了`train_batch_size` 外，复现时 **其他超参均与官方相同** 。

## 3. 训练模式

### 3.1 优化器

从 [tydi_modeling.py](https://github.com/google-research/language/blob/master/language/canine/tydiqa/tydi_modeling.py) 第 176 行可以定位到，canine 微调时候使用的优化器与 [bert 微调 tydi 优化器](https://github.com/google-research/language/blob/b76d2230156abec5c8d241073cdccbb36f66d1de/language/canine/bert_optimization.py#L24) 一致，为AdamW。因此本次复现采用相同的优化器和超参：

```python
optimizer = AdamWeightDecayOptimizer(
    learning_rate=learning_rate,
    weight_decay_rate=0.01,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
```

### 3.2 loss 计算

tydiQA 有两个不同的任务，官方选择对不同任务的损失取平均进行多任务训练。官方源码：`tydi_modeling.py` [link](https://github.com/google-research/language/blob/master/language/canine/tydiqa/tydi_modeling.py)；

本次复现的loss设计与官方相同，采用`answer_type`, `start_pos`, `end_pos` 三者 log softmax 的平均值来优化。`answer_type` 问题答案类型包括 passage，yes，no，None，minimal 5类。分别对应tydi任务中的 passage 任务，是非判断题，题目无答案，答案抽取情况。对于答案抽取任务，`start_pos`, `end_pos` 为答案的开头结尾位置。

## 4. 数据处理

本次复现使用的训练、测试数据处理方案与官方相同。但数据存储方式不同：1. 本次复现采用了 h5df 储存而非 tfrecord。2. 采用了list储存训练数据而非tf tensor。更多关于数据处理，请查看 `/tydi_canine/readme.md` 

## 5. 任务预测

相关任务预测代码基于 [canine/tydiqa](https://github.com/google-research/language/tree/b76d2230156abec5c8d241073cdccbb36f66d1de/language/canine/tydiqa) 做了内存占用及运算效率优化。除此外数据处理、模型测评的方法均与官方保持一致。

## 6. 模型笔记

模型要点：

+ tokenize free：模型采用 unicode 编码 + hash embedding，因此不像其他预训练模型一样需要vocabulary。
+ downsampling + upsampling：模型在transformer layer之前使用了卷积层来压缩输入序列长度。在支持输入序列长度 2048 的前提下，能保持和 mBert 相进的预测效率。

### 6.1 大致架构

#### 6.1.1 Embedding

> 输入 `[batch_size, len_seq]`
> 输出`[batch_size, len_seq, d_model]`

+ tokenizer 将原始输入直接转化为 unicode 整数。官方源码采用：`[ord(c) for c in text]` ；

+ embedding层采用了 hash embedding。

如果对所有 unicode 都安排一个 embedding，模型规模会爆炸。因此考虑采用 K 个大小为 `B * d/k` 的embedding矩阵来代替。主要思想来源于 [《Hash Embeddings for Efficient Word Representations》](http://arxiv.org/abs/1709.03933)。

```python
"num_hash_buckets": 16384,  # 论文公式中的 B
"num_hash_functions": 8,    # 官方采用的 num_hash_functions = K
"embedding_size": 768       # 论文公式中的 d
```

embedding 大致思路：

1. 有K 个 hash 方程，每个 hash 方程对应一个维度为 `[B,d/K]` 的可学习 embedding。
2. 将每个位置的 unicode 映射到 `B` 个区域 `hash buckets` 中：`hashed_id = ((input_ids + 1) * prime) % num_buckets`；这里的prime论文中给定了。
3. 根据 unicode 的 `hashed` 编码，从 K 个 embedding 矩阵中索引出该区域对应的表征（维度为 `[1, d/K]`）；然后拼接得到对应位置的embedding （`[1, d]`）。
4. 加上 position embedding，sentence type embedding 等。

因此，embedding部分的输入为 unicode 整数，输出是维度为 `[batch_size, len_seq, d_model]`  的embedding矩阵。此处 `len_seq = n = 2048`，论文中embedding公式：

![image-20220609204255117](img/note/image-20220609204255117.png)

#### 6.1.2 **down sampling** 

> 输入`[batch_size, len_seq, d_model]`
> 输出 `[batch_size, len_seq/down_sampling_rate, d_model]`
> canine-s中，`len_seq=2048`, `down_sampling_rate=4`

由于采用unicode编码导致句子长度变大，因此先对embedding进行一次down sampling。先进行local self attention，而后进行stride convolution。输出隐状态维度：`[batch_size, len_seq/r,d_model]` 

+ **local self attention（Single Local Transformer）**

论文中的 local self attention 大小为 128。即把embedding 分成大小为128的小embedding，而后分别进行self attention之后再进行拼接。此处有一个细节：论文对self attention做了小改动，来实现所有编码位置都能获取 cls 位置的注意力。具体表现为，在 `attention(q,k,v)` 中，将 cls 位置的表征拼接到 `v` 中。同时对 `q,k` 进行修改。输出 `h_init` 维度为 `[batch_size, len_seq, d]`

+ **strideconv**

采用了kernel为 4的1d conv进行压缩。特殊的是CLS位置编码需要保留。输出 `h_down = [batch_size, len_seq/down_sampling_rate, d_model]`

![image-20220609204310890](img/note/image-20220609204310890.png)

#### 6.1.3 Deep transformer stack

> 输入`[batch_size, len_seq/down_sampling_rate, d_model]`
> 输出 `[batch_size, len_seq/down_sampling_rate, d_model]`

深度编码模块。类似于bert的核心部分，即L层 transformer模块。对于非序列生成任务，文中直接使用CLS编码对应的隐状态进行预测。对于序列生成任务，还需要进行额外的解码步骤。该环节 `h_down` 规模同 down sampling，为 `[batch_size,len_seq/r, d]`

![image-20220609204324626](img/note/image-20220609204324626.png)

#### 6.1.4 Upsampling

> 输入`[batch_size, len_seq/down_sampling_rate, d_model]`
> 输出 `[batch_size, len_seq, d_model]`

upsampling 采用 `torch.repeat_interleave` ，将 `h_down` 维度转化为 `len_seq, d`。而后与 `h_init`拼接得到 `[len_seq, 2d]` 矩阵，最后通过 kernel 为 4 的 1d CONV，映射为 `len_seq, d`

 最后使用1层传统的transformer，输出结果 $y_{seq}$ 维度为 `[batch_size, len_seq, d]`

![image-20220609204334679](img/note/image-20220609204334679.png)

