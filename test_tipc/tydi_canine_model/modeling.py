from canine import CaninePretrainedModel
import paddle
import paddle.nn as nn

__all__ = ['CanineForTydiQA']


class CanineForTydiQA(CaninePretrainedModel):
    """
    Canine Model with special layers for TydiQA task.
    """

    def __init__(self, canine):
        super(CanineForTydiQA, self).__init__()
        self.canine = canine

        # dense layer for generate start, end prediction
        self.span_classifier = nn.Linear(
            in_features=self.canine.config["d_model"],
            out_features=2,
            weight_attr=paddle.framework.ParamAttr(
                name="span_linear_weight",
                learning_rate=1e-3,
                initializer=paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.02)),
            bias_attr=nn.initializer.Constant(value=0.0)
        )

        # there are 5 type of answers in tydiQA, the 5 is hard coded.
        self.answer_type_classifier = nn.Linear(
            in_features=self.canine.config["d_model"],
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