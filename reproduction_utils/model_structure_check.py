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
To check model forward propagation runs without bug
"""
import paddle
from canine import CanineTokenizer as PDTokenizer
from canine import CanineModel as PDmodel

paddle.set_device("cpu")

text = ["Life is."]


def run_check():
    model_name = "canine-s"
    PDtokenizer = PDTokenizer.from_pretrained(model_name)
    inputs = PDtokenizer(text,
                         padding="longest",
                         return_attention_mask=True,
                         return_token_type_ids=False)

    pd_inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}

    pd_model = PDmodel()
    pd_model.eval()

    with paddle.no_grad():
        outputs = pd_model(**pd_inputs)[0]

    print("outputs:", outputs[:, 0, :10])


if __name__ == "__main__":
    run_check()
