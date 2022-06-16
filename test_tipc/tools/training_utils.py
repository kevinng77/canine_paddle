import paddle

# Config file for tydiqa h5df dataset

# h5df
# |-- features: +-None, 3, 2048;  `input_ids, input_mask, segment_ids'
# |-- labels: None, 3; `start_pos, end_pos, answer_type`
# |-- metas: None, 3; ``unique_id, example_id, language_id`
# |-- offsets: None, 2, 2048; `start_offset, end_offset`

# define h5df group name
feature_group_name = 'features'
label_group_name = 'labels'
meta_group_name = 'metas'
offset_group_name = 'offsets'

# define shapes of h5 datasets
data_shapes = {feature_group_name: [1, 3, 2048],
               label_group_name: [1, 3],
               meta_group_name: [1, 3],
               offset_group_name: [1, 2, 2048]}


class CrossEntropyLossForTydi(paddle.nn.Layer):
    """
    Construct Loss Layers for TydiQA Minimum Answer Span task and Answer type prediction task.
    """

    def __init__(self):
        super(CrossEntropyLossForTydi, self).__init__()

    def forward(self, logits, start_positions, end_positions, answer_types):
        start_logits, end_logits, answer_type_logits = logits

        start_loss = paddle.nn.functional.cross_entropy(
            input=start_logits, label=start_positions)

        end_loss = paddle.nn.functional.cross_entropy(
            input=end_logits, label=end_positions)

        answer_types_loss = paddle.nn.functional.cross_entropy(
            input=answer_type_logits, label=answer_types)
        loss = (start_loss + end_loss + answer_types_loss) / 3
        return loss
