# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import os
import sys

paddle.set_device("cpu")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tydi_canine_model import CanineForTydiQA, CanineTokenizerforQA


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)
    parser.add_argument(
        '--save_inference_dir', default='./canine_infer', help='path where to save')
    parser.add_argument(
        '--model_path', default="canine-s-tydiqa-finetuned", help='path to pretrained model')
    args = parser.parse_args()
    return args


def export(args):
    # build model
    model = CanineForTydiQA.from_pretrained(args.model_path)
    tokenizer = CanineTokenizerforQA()  # TODO update QA tokenizer
    model.eval()
    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name="input_ids"
            ),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name="token_type_ids"),  # required for MRC task
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name="attention_mask"),
        ],
    )
    # save inference_python model
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference_python"))
    tokenizer.save_pretrained(args.save_inference_dir)
    print(
        f">>> inference_python model and tokenizer have been saved into {args.save_inference_dir}"
    )


if __name__ == "__main__":
    args = get_args()
    export(args)