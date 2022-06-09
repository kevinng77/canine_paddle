# repeat_interleave 算子

paddlepaddle==2.3.0 中该算子似乎有问题，但目前没定位到问题所在。

采用了 `repeat_interleave` 算子的模型前项传导没问题，但在反向传播是有一定几率出现 segmentation fault (core)。

### 错误示范

在反向传播时候，采用 `paddle.repeat_interleave`的 `repeat_molecules` 对于相同的输入，都会有不同的几率出现 `segmentation fault`

```python
def repeat_molecules(molecules, char_seq_length:int,repeat_func:callable):
    """Repeats molecules to make them the same length as the char sequence."""

    rate = 2
    molecules_without_extra_cls = molecules[:, 1:, :]
    repeated = repeat_func(molecules_without_extra_cls, repeats=rate, axis=-2)
    last_molecule = molecules[:, -1:, :]
    remainder_length = paddle.mod(paddle.to_tensor(char_seq_length), paddle.to_tensor(rate)).item()
    remainder_repeated = repeat_func(
        last_molecule,
        repeats=remainder_length + rate,
        axis=-2,
    )
    return paddle.concat([repeated, remainder_repeated], axis=-2)

# 个人采用其他方式暂时取代了 `paddle.repeat_interleave`，代替后，模型经过3天的训练测试一切正常。如下：
def repeat_interleave(input, repeats, axis=-2):
    # repeat interleave for canine model only!! (avoid paddle.interleave .backward() bug)
    # only for repeat on axis=-2
    batch_size, len_seq, d_model = input.shape
    flat_input = paddle.tile(input, repeat_times=(1, 1, repeats))
    return paddle.reshape(flat_input, [batch_size, -1, d_model])


def main():
    hidden_size = 256
    dense = paddle.nn.Linear(hidden_size,1)
    print("start testing")
    fixed_input = paddle.randn([2,512,hidden_size])
    a = repeat_molecules(fixed_input, char_seq_length=2048, repeat_func=paddle.repeat_interleave)
    b = repeat_molecules(fixed_input,char_seq_length=2048,repeat_func=repeat_interleave)
    print("operator difference",paddle.sum(a-b).item())
    for i in range(10000):
        if i % 100 == 0:
            print(f"pass {i} iteration")
        i += 1
        input = fixed_input
        # repeated = repeat_molecules(input,char_seq_length=2048,repeat_func=repeat_interleave)
        repeated = repeat_molecules(input,char_seq_length=2048,repeat_func=paddle.repeat_interleave)
        reduced = paddle.mean(repeated,axis=-2)
        output = dense(reduced)
        loss = paddle.sum(output)
        loss.backward()
    print("done without bug")
    return
```

报错信息（CPU运行）：

```shell
(python37) ~/shared/canine_paddle python test.py
start testing
operator difference 0.0
pass 0 iteration
pass 100 iteration
pass 200 iteration


--------------------------------------
C++ Traceback (most recent call last):
--------------------------------------
No stack trace in paddle, may be caused by external reasons.

----------------------
Error Message Summary:
----------------------
FatalError: `Segmentation fault` is detected by the operating system.
  [TimeInfo: *** Aborted at 1653890794 (unix time) try "date -d @1653890794" if you are using GNU date ***]
  [SignalInfo: *** SIGSEGV (@0x0) received by PID 1216 (TID 0x7f0e17b94180) from PID 0 ***]

[1]    1216 segmentation fault  python test.py
```

报错信息（GPU运行）

```shell
----------------------
Error Message Summary:
----------------------
FatalError: `Segmentation fault` is detected by the operating system.
  [TimeInfo: *** Aborted at 1653889744 (unix time) try "date -d @1653889744" if you are using GNU date ***]
  [SignalInfo: *** SIGSEGV (@0x0) received by PID 529 (TID 0x7f46c6123740) from PID 0 ***]

Segmentation fault
```

