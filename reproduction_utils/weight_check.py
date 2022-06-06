#encoding=utf8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
generate files that maps torch layer weight to paddle layers for converting weight.
"""
import torch
from collections import OrderedDict
import paddle
from paddlenlp.transformers import CanineModel as PDmodel


def weight_check(pytorch_checkpoint_path,
                 paddle_dump_path,
                 mapping_file="./torch_paddle_layer_map.json"):
    # pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    pd_model = PDmodel.from_pretrained('canine-s')
    paddle_state_dict = paddle.load(paddle_dump_path)
    loaded_state_dict = pd_model.state_dict()
    paddle.set_device("cpu")

    for k,v in paddle_state_dict.items():
        diff = paddle.mean(paddle.cast(v - loaded_state_dict[k],'float64')).numpy()[0]
        if abs(diff) > 1e-6:
            print(f"difference:\t {diff:.5f}")
            if v.ndim == 1:
                print(f"{k}\ntarget",v.numpy()[:5])
                print(f"loaded",loaded_state_dict[k].numpy()[:5])
            else:
                print(f"{k}\ntarget",v.numpy()[0,:5])
                print(f"loaded",loaded_state_dict[k].numpy()[0,:5])



if __name__ == "__main__":
    weight_check(pytorch_checkpoint_path="torch_weight/pytorch_model.bin",
                paddle_dump_path="../data/checkout_point/model_state.pdparams")