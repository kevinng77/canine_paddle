# CANINE Test TIPC

本环节中，canine模型将采用在 TydiQA 任务上微调的模型权重 [canine-s-tydiqa-finetuned](https://huggingface.co/kevinng77/paddle-tydiQA-canine-s/blob/main/model_state.pdparams) 进行 TIPC 测试。使用 `.from_pretrained("canine-s-tydiqa-finetuned")` 会自动下载权重。

## 目录
- [CANINE Test TIPC](#canine-test-tipc)
  - [目录](#目录)
  - [简介](#简介)
  - [开始使用](#开始使用)
    - [基础使用](#基础使用)
    - [模型预测](#模型预测)
  - [模型推理部署](#模型推理部署)
    - [基于Inference的推理](#基于inference的推理)
    - [基于Serving的服务化部署](#基于serving的服务化部署)
  - [TIPC自动化测试脚本](#tipc自动化测试脚本)
  - [LICENSE](#license)
## 简介

世界上存在海量的语言与词汇，在处理多语言场景时，传统预训练模型采用的 Vocab 和 Tokenization 方案难免会遇到 out of vocabulary 和 unkonw token 的情况。
Canine 提供了 tokenization-free 的预训练模型方案，提高了模型在多语言任务下的能力。

论文链接：[CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://paperswithcode.com/paper/canine-pre-training-an-efficient-tokenization)
参考repo：[google-research/language](https://github.com/google-research/language/tree/master/language/canine)，[huggingface/transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/canine)。

关于 canine paddle 模型在 TydiQA 阅读理解任务上的复现情况，请参考该仓库根目录。

## 开始使用

环境准备

```
paddlepaddl==2.3.0
reprod_log 
paddlenlp==2.3.1
```

### 基础使用

```python
from canine import CanineTokenizer
from canine import CanineModel

tokenizer = CanineTokenizer.from_pretrained("canine-s")
model = CanineModel.from_pretrained("canine-s")
text = ["canine is tokenization-free"]

inputs = tokenizer(text,
                     padding="longest",
                     return_attention_mask=True,
                     return_token_type_ids=True, )
pd_inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
seq_outputs, pooling_outputs = model(**pd_inputs)
```

### 模型预测

阅读理解预测示例：

```python
test_article = "The Nobel Prize in Literature (Swedish: Nobelpriset i litteratur) is awarded annually by the Swedish Academy to authors for outstanding contributions in the field of literature. It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, which are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.[1] As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Swedish Academy.[2] The first Nobel Prize in Literature was awarded in 1901 to Sully Prudhomme of France.[3] Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.[4] In 1901, Prudhomme received 150,782 SEK, which is equivalent to 8,823,637.78 SEK in January 2018."

test_question = "Who was the first Nobel prize winner for Literature?"
```

- 使用GPU预测

```shell
python -m tools.predict --model_dir=canine-s-tydiqa-finetuned --use_gpu=1
```

终端输出：`>>> Answer Text: Sully Prudhomme of France, score: 4.681941032409668`。表示阅读理解的答案为 `Sully Prudhomme of France`，置信度 `4.681941032409668`

- 使用CPU预测

```shell
python -m tools.predict --model_dir=canine-s-tydiqa-finetuned
```

终端输出：`>>> Answer Text: Sully Prudhomme of France, score: 4.677703857421875`。表示阅读理解的答案为 `Sully Prudhomme of France`，置信度 `4.677703857421875`与GPU模型下有一定差距。

## 模型推理部署

### 基于Inference的推理

**模型动态转静态：**

使用下面的命令完成 `canine-s`模型的动转静导出，其中 `canine-s-tydiqa-finetuned` 为在 TydiQA 任务上微调后的权重 。

```shell
python tools/export_model.py --save_inference_dir=./canine_infer --model_path=canine-s-tydiqa-finetuned
```

最终在 `canine_infer/`文件夹下会生成3个 `inference.xxx`文件。

**模型推理：**

```shell
python deploy/inference_python/infer.py --model_dir=./canine_infer
```

在终端中输出结果如下。

```shell
>>> Article: The Nobel Prize in Literature (Swedish: Nobelpriset i litteratur) is awarded annually by the Swedish Academy to authors for outstanding contributions in the field of literature. It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, which are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.[1] As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Swedish Academy.[2] The first Nobel Prize in Literature was awarded in 1901 to Sully Prudhomme of France.[3] Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.[4] In 1901, Prudhomme received 150,782 SEK, which is equivalent to 8,823,637.78 SEK in January 2018.
>>> Question: Who was the first Nobel prize winner for Literature?

>>> Answer Text: Sully Prudhomme of France, score: 4.677705764770508
```

表示对于问题 "Who was the first Nobel prize winner for Literature?" 的答案是 "Sully Prudhomme of France"，置信度为 4.6777。该结果与基于训练引擎的cpu预测结果仅相差 `1e-6`。

【备注】详细请参考 [deploy/inference_python](deploy/inference_python) 文件夹。

### 基于Serving的服务化部署

## TIPC自动化测试脚本

```shell
bash test_tipc/test_train_inference_python.sh test_tipc/configs/canine/train_infer_python.txt lite_train_lite_infer
```

`output/result_python.log`输出结果如下，表示命令运行成功。

```shell
[33m Run successfully with command - python train.py --fp16 --input_dir="../data/tydi/train.h5df"    --output_dir=./output/norm_train_gpus_0_autocast_null --max_steps=100     --batch_size=2     !  [0m
[33m Run successfully with command - python tools/export_model.py --model_path=./canine_tydi_qa --save_inference_dir=./canine_infer      !  [0m
[33m Run successfully with command - python deploy/inference_python/infer.py --model_dir=./canine_infer --use_gpu=False --benchmark=False               > ./output/python_infer_cpu_usemkldnn_False_threads_null_precision_null_batchsize_null.log 2>&1 !  [0m
```

【备注】详细请参考 [test_tipc/test_tipc 文件夹](test_tipc/readme.md)

## LICENSE

本项目的发布受[Apache 2.0 license](https://github.com/JunnYu/xlm_paddle/blob/main/test_tipc/LICENSE)许可认证。
