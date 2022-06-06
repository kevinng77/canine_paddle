# 复现日志与结果整理

## 文件结构

```python
.
├── 3090_fp32_seed6                      # 每个文件夹对应一次复现结果（微调、预测、评估）
│   ├── eval_result.png                  # 运行 `tydi_eval.py` 后评估结果截图
│   ├── finetune_3090_fp32_seed6.log     # 微调日志
│   ├── prediction_3090_fp32_seed6.log   # 预测日志
│   └── result_3090_fp32_seed6.jsonl     # `tydi_eval.py` 评估时所需要的 jsonl 文件
├── readme.md
├── v100_fp16_seed2020
├── v100_fp16_seed2021
├── v100_fp16_seed2022
├── v100_fp16_seed6
└── v100_fp16_seed666

```

## 其他

+ `eval_result.png` 和 `result_***.jsonl` 的关系：

  `result_***.jsonl` 为格式满足 TydiQA 评测要求的文件。在根目录执行：

```shell
python3 official_tydi/tydi_eval.py \
  --gold_path=data/tydi/tydiqa-v1.0-dev.jsonl.gz \
  --predictions_path=logs/3090_fp32_seed6/result_3090_fp32_seed6.jsonl
```

TydiQA 跑分程序会将结果输出到终端中，`eval_result.png`即为输出结果的截图。