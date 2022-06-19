import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from canine import CaninePretrainedModel
import paddle
import paddle.nn as nn

__all__ = ['CanineForTydiQA']


class CanineForTydiQA(CaninePretrainedModel):
    """
    Canine Model with special layers for TydiQA task.
    """
    base_model_prefix = "canine"
    pretrained_init_configuration = {
        "canine-s": {
            "hidden_size": 768,
            "bos_token_id": 57344,
            "eos_token_id": 57345,
            "pad_token_id": 0,
            "num_encoder_layers": 12,
            "num_heads": 12,
            "activation": "gelu",
            "num_hash_buckets": 16384,
            "num_hash_functions": 8,
            "type_vocab_size": 16,
            "layer_norm_eps": 1e-12,
            "upsampling_kernel_size": 4,
            "downsampling_rate": 4,
            "local_transformer_stride": 128,
            "max_position_embeddings": 16384,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "encoder_ffn_dim": 3072,
            "init_std": 0.02,
            'model_max_length': 2048
        },
        "canine-s-tydiqa-finetuned":{
            "hidden_size": 768,
            "bos_token_id": 57344,
            "eos_token_id": 57345,
            "pad_token_id": 0,
            "num_encoder_layers": 12,
            "num_heads": 12,
            "activation": "gelu",
            "num_hash_buckets": 16384,
            "num_hash_functions": 8,
            "type_vocab_size": 16,
            "layer_norm_eps": 1e-12,
            "upsampling_kernel_size": 4,
            "downsampling_rate": 4,
            "local_transformer_stride": 128,
            "max_position_embeddings": 16384,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "encoder_ffn_dim": 3072,
            "init_std": 0.02,
            'model_max_length': 2048
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "canine-s":
            # TODO edit weight path
                "https://huggingface.co/kevinng77/paddle-canine-s/resolve/main/model_state.pdparams",
            # add a finetuned weight link for test tipc
            "canine-s-tydiqa-finetuned":
                "https://huggingface.co/kevinng77/paddle-tydiQA-canine-s/resolve/main/model_state.pdparams"
        }
    }

    def __init__(self, canine):
        super(CanineForTydiQA, self).__init__()
        self.canine = canine

        # dense layer for generate start, end prediction
        self.span_classifier = nn.Linear(
            in_features=self.canine.config["hidden_size"],
            out_features=2,
            weight_attr=paddle.framework.ParamAttr(
                name="span_linear_weight",
                learning_rate=1e-3,
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.02)),
            bias_attr=nn.initializer.Constant(value=0.0)
        )

        # there are 5 type of answers in tydiQA, the 5 is hard coded.
        self.answer_type_classifier = nn.Linear(
            in_features=self.canine.config["hidden_size"],
            out_features=5,
            weight_attr=paddle.framework.ParamAttr(
                name="answer_type_linear_weight",
                learning_rate=1e-3,
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.02)),
            bias_attr=nn.initializer.Constant(value=0.0))

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None):
        sequence_output, pooling_output = self.canine(input_ids=input_ids,
                                                      token_type_ids=token_type_ids,
                                                      attention_mask=attention_mask
                                                      )

        logits = self.span_classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        answer_type_logits = self.answer_type_classifier(pooling_output)

        return start_logits, end_logits, answer_type_logits