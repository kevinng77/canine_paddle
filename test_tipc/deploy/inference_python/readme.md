# Inference 推理

# 目录
- [Inference 推理](#inference-推理)
- [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 推理过程](#2-推理过程)
    - [2.1 准备推理环境](#21-准备推理环境)
    - [2.2 模型动转静导出](#22-模型动转静导出)
    - [2.3 模型推理](#23-模型推理)
## 1. 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。

本文档主要基于Paddle Inference的 canine-s 模型推理。

更多关于Paddle Inference推理引擎的介绍，可以参考[Paddle Inference官网教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/inference_cn.html)。

## 2. 推理过程

### 2.1 准备推理环境

安装好PaddlePaddle即可体验Paddle Inference部署能力。

### 2.2 模型动转静导出

使用下面的命令完成`canine-s`模型的动转静导出。

```shell
python -m tools.export_model --save_inference_dir=./canine_infer --model_path=./canine_tydi_qa
```

最终在`canine_infer/`文件夹下会生成下面的3个文件。

```txt
canine_infer
     |----inference.pdiparams     : 模型参数文件
     |----inference.pdmodel       : 模型结构文件
     |----inference.pdiparams.info: 模型参数信息文件
```

### 2.3 模型推理

```shell\
python -m deploy.inference_python.infer --model_dir=./canine_infer
```

对于下面的文章和问题进行阅读理解预测：

```python
test_article = "The Nobel Prize in Literature (Swedish: Nobelpriset i litteratur) is awarded annually by the Swedish Academy to authors for outstanding contributions in the field of literature. It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, which are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.[1] As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Swedish Academy.[2] The first Nobel Prize in Literature was awarded in 1901 to Sully Prudhomme of France.[3] Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.[4] In 1901, Prudhomme received 150,782 SEK, which is equivalent to 8,823,637.78 SEK in January 2018."

test_question = "Who was the first Nobel prize winner for Literature?"
```

在终端中输出结果如下。

```shell
>>> Article: The Nobel Prize in Literature (Swedish: Nobelpriset i litteratur) is awarded annually by the Swedish Academy to authors for outstanding contributions in the field of literature. It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, which are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.[1] As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Swedish Academy.[2] The first Nobel Prize in Literature was awarded in 1901 to Sully Prudhomme of France.[3] Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.[4] In 1901, Prudhomme received 150,782 SEK, which is equivalent to 8,823,637.78 SEK in January 2018.
>>> Question: Who was the first Nobel prize winner for Literature?

>>> Answer Text: Sully Prudhomme of France, score: 4.677705764770508
```

表示对于问题 "Who was the first Nobel prize winner for Literature?" 的答案是 "Sully Prudhomme of France"，置信度为 4.678。该结果与基于训练引擎的结果完全一致。