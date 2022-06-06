# coding=utf-8
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 Google AI The HuggingFace Inc. team. All rights reserved.
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
import paddle.nn as nn
import paddle.tensor as tensor
from paddle.fluid.data_feeder import convert_dtype
from paddle.nn import functional as F
from paddle.fluid import layers
from paddle.fluid.dygraph import LayerList

from paddlenlp.transformers import PretrainedModel, register_base_model

__all__ = [
    'CanineModel', 'CaninePretrainedModel', 'CanineEmbeddings'
]

# Support up to 16 hash functions.
_PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]
dtype_float = paddle.get_default_dtype()


def _convert_attention_mask(attn_mask, dtype):
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            # attention mask value is modified to -10000
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 10000
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class CaninePretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained Canine models. It provides Canine related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models.
    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """
    base_model_prefix = "canine"
    pretrained_init_configuration = {
        "canine-s": {
            "d_model": 768,
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
        },
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "canine-s":
            # TODO edit weight path
                "data/paddle_weight/model_state.pdparams",
        }
    }

    def init_weights(self, layer):
        """ Initialization hook """
        if paddle.get_default_dtype() not in ['float32', 'float64']:
            # gaussian/standard_normal/randn/normal only supports [float32, float64]
            return
        if isinstance(layer, (nn.Linear, nn.Embedding, nn.Conv1D)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.init_std if hasattr(self, "init_std") else
                        self.canine.config["init_std"],
                        shape=layer.weight.shape))


class CanineEmbeddings(nn.Layer):
    """
    Construct Embedding Layer for Canine Model based on Character hash embeddings.

    Please refer to :class:`~paddlenlp.transformers.Canine.CanineModel` for more information regarding the arguments.
    Please refer to `https://paperswithcode.com/paper/canine-pre-training-an-efficient-tokenization` for
    more about Character Hash Embeddings.
    """

    def __init__(self,
                 embedding_dim,
                 num_hash_functions=8,
                 num_hash_buckets=16384,
                 type_vocab_size=16,
                 layer_norm_eps=1e-12,
                 dropout=0.1,
                 max_position_embeddings=16384,
                 ):
        super(CanineEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_hash_functions = num_hash_functions
        self.num_hash_buckets = num_hash_buckets

        shard_embedding_size = embedding_dim // num_hash_functions

        for i in range(num_hash_functions):
            name = f"HashBucketCodepointEmbedder_{i}"
            setattr(self, name, nn.Embedding(num_embeddings=num_hash_buckets,
                                             embedding_dim=shard_embedding_size))

        self.char_position_embeddings = nn.Embedding(num_embeddings=num_hash_buckets,
                                                     embedding_dim=embedding_dim)
        self.token_type_embeddings = nn.Embedding(num_embeddings=type_vocab_size,
                                                  embedding_dim=embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("position_ids", paddle.arange(max_position_embeddings, dtype="int64").expand((1, -1)))

    # Copied from google-research/language/blob/master/language/canine/modeling.py _hash_bucket_tensors
    def _hash_bucket_tensors(self,
                             input_ids,
                             num_hashes,
                             num_buckets):
        """Converts ids to hash bucket ids via multiple hashing.

        Args:
            input_ids (Tensor): The codepoints or other IDs to be hashed.
            num_hashes (int): The number of hash functions to use.
            num_buckets (int): The number of hash buckets (i.e. embeddings in each table).

        Returns:
            List: A list of tensors, each of which is the hash bucket IDs from one hash function.
        """
        if num_hashes > len(_PRIMES):
            raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

        primes = _PRIMES[:num_hashes]

        result_tensors = []
        for prime in primes:
            hashed = ((input_ids + 1) * prime) % num_buckets
            result_tensors.append(hashed)
        return result_tensors

    # Copied from google-research/language/blob/master/language/canine/modeling.py _embed_hash_buckets
    def _embed_hash_buckets(self,
                            input_ids,
                            embedding_dim,
                            num_hashes,
                            num_buckets):
        """
        Convert input ids (Unicode code points) into embeddings based on concatenating multiple hash embeddings.

        Args:
            input_ids (Tensor): The codepoints or other IDs to be hashed.
            embedding_dim (int): The dimension of embedding.
            num_hashes (int): The number of hash functions to use.
            num_buckets (int): The number of hash buckets (i.e. embeddings in each table).

        Returns:
            Tensor: Hashed Embeddings of input ids.
        """
        if embedding_dim % num_hashes != 0:
            raise ValueError(f"Expected `embedding_size` ({embedding_dim}) % `num_hashes` ({num_hashes}) == 0")

        hash_bucket_tensors = self._hash_bucket_tensors(input_ids=input_ids,
                                                        num_hashes=num_hashes,
                                                        num_buckets=num_buckets)

        embedding_shards = []
        for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
            name = f"HashBucketCodepointEmbedder_{i}"
            shard_embeddings = getattr(self, name)(hash_bucket_ids)
            embedding_shards.append(shard_embeddings)

        return paddle.concat(embedding_shards, axis=-1)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self._embed_hash_buckets(
                input_ids, self.embedding_dim, self.num_hash_functions, self.num_hash_buckets
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        # canine use absolute position embedding
        position_embeddings = self.char_position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

        embeddings = self.dropout(self.layer_norm(embeddings))
        return embeddings


class CharactersToMolecules(nn.Layer):
    """
    Construct down sampling Layer for Canine Model.

    Please refer to :class:`~paddlenlp.transformers.Canine.CanineModel` for more information regarding the arguments.
    Please refer to `https://paperswithcode.com/paper/canine-pre-training-an-efficient-tokenization` for
    more about down sampling.
    """

    def __init__(self,
                 hidden_size=768,
                 down_sampling_rate=4,
                 activation='gelu',
                 layer_norm_eps=1e-12
                 ):
        super(CharactersToMolecules, self).__init__()
        self.conv = nn.Conv1D(in_channels=hidden_size,
                              out_channels=hidden_size,
                              kernel_size=down_sampling_rate,
                              stride=down_sampling_rate,
                              )

        self.activation = getattr(F, activation)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self, char_encoding):
        """
        Args:
            char_encoding (Tensor) : hidden states with shape `[batch_size, char_seq_length, hidden_size]`
        Return:
            Tensor: output hidden states with shape `[batch_size, molecule_seq_length, hidden_size]`
                    , where `molecule_seq_length == char_seq_length/down_sampling_rate`
        """
        cls_encoding = char_encoding[:, 0:1, :]  # `cls_encoding`: [batch, 1, hidden_size]

        # Transpose char_encoding to be [batch, hidden_size, char_seq]
        char_encoding = paddle.transpose(char_encoding, [0, 2, 1])
        downsampled = self.conv(char_encoding)
        downsampled = paddle.transpose(downsampled, [0, 2, 1])
        downsampled = self.activation(downsampled)

        # Truncate the last molecule in order to reserve a position for [CLS].
        # Often, the last position is never used (unless we completely fill the
        # text buffer). This is important in order to maintain alignment on TPUs
        # (i.e. a multiple of 128).
        downsampled_truncated = downsampled[:, 0:-1, :]

        # We also keep [CLS] as a separate sequence position since we always
        # want to reserve a position (and the model capacity that goes along
        # with that) in the deep BERT stack.
        result = paddle.concat([cls_encoding, downsampled_truncated], axis=1)

        result = self.layer_norm(result)

        return result


class ConvProjection(nn.Layer):
    """
    Construct up sampling Layer for Canine Model.

    Please refer to :class:`~paddlenlp.transformers.Canine.CanineModel` for more information regarding the arguments.
    Please refer to `https://paperswithcode.com/paper/canine-pre-training-an-efficient-tokenization` for
    more about up sampling.
    """

    def __init__(self,
                 hidden_size=768,
                 upsampling_kernel_size=4,
                 activation='gelu',
                 hidden_dropout=0.1,
                 layer_norm_eps=1e-12
                 ):
        super(ConvProjection, self).__init__()
        self.conv = nn.Conv1D(in_channels=hidden_size * 2,
                              out_channels=hidden_size,
                              kernel_size=upsampling_kernel_size,
                              stride=1,
                              )
        self.upsampling_kernel_size = upsampling_kernel_size
        self.activation = getattr(F, activation)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): hidden states with shape `[batch_size, char_seq_length, 2 * hidden_size]`

        Returns:
            Tensor: up sampled hidden states with shape `[batch_size, char_seq_length, hidden_size]`
        """
        # Transpose inputs to be [batch, 2 * hidden_size, molecule_seq_length]
        inputs = paddle.transpose(inputs, [0, 2, 1])
        pad_total = self.upsampling_kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        pad = nn.Pad1D(padding=(pad_beg, pad_end),
                       mode='constant',
                       value=0)

        result = self.conv(pad(inputs))
        result = paddle.transpose(result, [0, 2, 1])
        result = self.dropout(self.layer_norm(self.activation(result)))

        # TODO add support for MLM task
        return result


class CanineSelfAttention(nn.MultiHeadAttention):
    """
    Construct self Attention Layer for Canine Model to support Single Local Attention and Traditional Attention.

    Please refer to :class:`~paddlenlp.transformers.MultiHeadAttention` regarding details of parameters.
    """

    def __init__(self, embed_dim=768,
                 num_heads=12,
                 dropout=0.1,
                 ):
        super(CanineSelfAttention, self).__init__(embed_dim=embed_dim,
                                                  num_heads=num_heads,
                                                  dropout=dropout, )

        # Owning to support single local attention, linear projection after
        # multi-head attention is moved to `~paddlenlp.transformers.Canine.CanineAttention`
        self.out_proj = None

    def forward(self,
                query,
                key=None,
                value=None,
                attn_mask=None,
                head_mask=None,
                cache=None):
        """
        Overwrite `~paddlenlp.transformers.MultiHeadAttention.forward` to support Single Local Attention.
        Please refer to :class:`~paddlenlp.transformers.MultiHeadAttention` regarding parameters.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        # TODO: use tensor.matmul, however it doesn't support `alpha`
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim ** -0.5)

        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = paddle.unsqueeze(attn_mask, axis=1)
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask

        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        # masked head for cls token in Canine
        if head_mask is not None:
            weights = weights * head_mask

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class CanineAttention(nn.Layer):
    """
    Construct Attention Layer for Canine Model to support Single Local Attention and Traditional Attention.

    Please refer to :class:`~paddlenlp.transformers.Canine.CanineModel` for more information regarding the arguments.

    """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 attn_dropout=0.1,
                 hidden_dropout=0.1,
                 layer_norm_eps=1e-12,
                 local=False,
                 always_attend_to_first_position=False,  # TODO check parameter usage
                 first_position_attends_to_all=False,  # TODO check parameter usage
                 attend_from_chunk_width=128,
                 attend_from_chunk_stride=128,
                 attend_to_chunk_width=128,
                 attend_to_chunk_stride=128,
                 ):
        super(CanineAttention, self).__init__()

        self.self_attn = CanineSelfAttention(embed_dim=hidden_size,
                                             num_heads=num_heads,
                                             dropout=attn_dropout,
                                             )

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout)

        self.local = local
        if attend_from_chunk_width < attend_from_chunk_stride:
            raise ValueError(
                "`attend_from_chunk_width` < `attend_from_chunk_stride` would cause sequence positions to get skipped."
            )
        if attend_to_chunk_width < attend_to_chunk_stride:
            raise ValueError(
                "`attend_to_chunk_width` < `attend_to_chunk_stride`would cause sequence positions to get skipped."
            )
        self.always_attend_to_first_position = always_attend_to_first_position
        self.first_position_attends_to_all = first_position_attends_to_all
        self.attend_from_chunk_width = attend_from_chunk_width
        self.attend_from_chunk_stride = attend_from_chunk_stride
        self.attend_to_chunk_width = attend_to_chunk_width
        self.attend_to_chunk_stride = attend_to_chunk_stride

    def forward(self,
                hidden_states,
                attn_mask=None,
                head_mask=None
                ):
        if not self.local:
            attention_output = self.self_attn(query=hidden_states,
                                              attn_mask=attn_mask,
                                              head_mask=head_mask)
        else:
            # Perform Single Local Attention
            from_seq_length = to_seq_length = hidden_states.shape[1]
            from_tensor = to_tensor = hidden_states

            # Create chunks (windows) that we will attend *from* and then concatenate them.
            from_chunks = []
            if self.first_position_attends_to_all:
                from_chunks.append((0, 1))
                # We must skip this first position so that our output sequence is the
                # correct length (this matters in the *from* sequence only).
                from_start = 1
            else:
                from_start = 0
            for chunk_start in range(from_start, from_seq_length, self.attend_from_chunk_stride):
                chunk_end = min(from_seq_length, chunk_start + self.attend_from_chunk_width)
                from_chunks.append((chunk_start, chunk_end))

            # Determine the chunks (windows) that will will attend *to*.
            to_chunks = []
            if self.first_position_attends_to_all:
                to_chunks.append((0, to_seq_length))
            for chunk_start in range(0, to_seq_length, self.attend_to_chunk_stride):
                chunk_end = min(to_seq_length, chunk_start + self.attend_to_chunk_width)
                to_chunks.append((chunk_start, chunk_end))

            if len(from_chunks) != len(to_chunks):
                raise ValueError(
                    f"Expected to have same number of `from_chunks` ({from_chunks}) and "
                    f"`to_chunks` ({from_chunks}). Check strides."
                )

            # compute attention scores for each pair of windows and concatenate the results.
            attention_output_chunks = []
            for (from_start, from_end), (to_start, to_end) in zip(from_chunks, to_chunks):
                from_tensor_chunk = from_tensor[:, from_start:from_end, :]
                to_tensor_chunk = to_tensor[:, to_start:to_end, :]
                # `attention_mask`: <float>[batch_size, from_seq, to_seq]
                # `attention_mask_chunk`: <float>[batch_size, from_seq_chunk, to_seq_chunk]
                attention_mask_chunk = attn_mask[:, from_start:from_end, to_start:to_end]
                if self.always_attend_to_first_position:
                    cls_attention_mask = attn_mask[:, from_start:from_end, 0:1]
                    attention_mask_chunk = paddle.concat([cls_attention_mask, attention_mask_chunk],
                                                         axis=2)

                    cls_position = to_tensor[:, 0:1, :]
                    to_tensor_chunk = paddle.concat([cls_position, to_tensor_chunk],
                                                    axis=1)

                attention_outputs_chunk = self.self_attn(query=from_tensor_chunk,
                                                         key=to_tensor_chunk,
                                                         value=to_tensor_chunk,
                                                         attn_mask=attention_mask_chunk,
                                                         head_mask=head_mask,
                                                         )
                attention_output_chunks.append(attention_outputs_chunk)

            attention_output = paddle.concat(attention_output_chunks, axis=1)

        attention_output = self.dense(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(hidden_states + attention_output)

        return attention_output


class CanineLayer(nn.Layer):
    """
    Construct Canine Transformer Layer, supporting Single Local Attention and Traditional Attention.

    Please refer to :class:`~paddlenlp.transformers.Canine.CanineModel` for more information regarding the arguments.
    """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 ffn_dim,
                 activation='gelu',
                 layer_norm_eps=1e-12,
                 attn_dropout=0.1,
                 hidden_dropout=0.1,
                 local=False,
                 always_attend_to_first_position=False,  # TODO check parameter usage
                 first_position_attends_to_all=False,
                 attend_from_chunk_width=128,
                 attend_from_chunk_stride=128,
                 attend_to_chunk_width=128,
                 attend_to_chunk_stride=128,
                 ):
        super(CanineLayer, self).__init__()
        self.seq_len_dim = 1
        self.attention = CanineAttention(hidden_size=hidden_size,
                                         num_heads=num_heads,
                                         attn_dropout=attn_dropout,
                                         hidden_dropout=hidden_dropout,
                                         layer_norm_eps=layer_norm_eps,
                                         local=local,
                                         always_attend_to_first_position=always_attend_to_first_position,
                                         first_position_attends_to_all=first_position_attends_to_all,
                                         attend_from_chunk_width=attend_from_chunk_width,
                                         attend_from_chunk_stride=attend_from_chunk_stride,
                                         attend_to_chunk_width=attend_to_chunk_width,
                                         attend_to_chunk_stride=attend_to_chunk_stride,
                                         )

        self.ffn = nn.Linear(in_features=hidden_size, out_features=ffn_dim)

        self.ffn_activation = getattr(F, activation)
        self.ffn_output = nn.Linear(ffn_dim, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)

    def forward(self,
                hidden_states,
                attn_mask=None,
                head_mask=None,
                ):
        # TODO will support cache?
        self_attention_outputs = self.attention(
            hidden_states,
            attn_mask,
            head_mask,
        )
        hidden_state = self.ffn_activation(self.ffn(self_attention_outputs))

        hidden_state = self.ffn_output(hidden_state)
        hidden_state = self.dropout(hidden_state)

        hidden_state = self.layer_norm(hidden_state + self_attention_outputs)

        return hidden_state


class CanineEncoder(nn.Layer):
    """
    Construct Encoder module for Canine Model.

    Please refer to :class:`~paddlenlp.transformers.Canine.CanineModel` for more information
    regarding the arguments.
    """

    # TODO if head mask never used, this class could be replaced by nn.TransformerEncoder()

    def __init__(self,
                 hidden_size,
                 num_heads,
                 activation,
                 encoder_ffn_dim,
                 num_encoder_layers,
                 attn_dropout=0.1,
                 hidden_dropout=0.1,
                 layer_norm_eps=1e-12,
                 local=False,
                 always_attend_to_first_position=False,
                 first_position_attends_to_all=False,
                 attend_from_chunk_width=128,
                 attend_from_chunk_stride=128,
                 attend_to_chunk_width=128,
                 attend_to_chunk_stride=128,
                 ):

        super(CanineEncoder, self).__init__()
        self.layers = LayerList([
            CanineLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                activation=activation,
                local=local,
                ffn_dim=encoder_ffn_dim,
                layer_norm_eps=layer_norm_eps,
                attn_dropout=attn_dropout,
                hidden_dropout=hidden_dropout,
                always_attend_to_first_position=always_attend_to_first_position,
                first_position_attends_to_all=first_position_attends_to_all,
                attend_from_chunk_width=attend_from_chunk_width,
                attend_from_chunk_stride=attend_from_chunk_stride,
                attend_to_chunk_width=attend_to_chunk_width,
                attend_to_chunk_stride=attend_to_chunk_stride,
            )
            for _ in range(num_encoder_layers)])

    def forward(
            self,
            hidden_states,
            attn_mask=None,
            head_mask=None,  # TODO head mask can remove?
            cache=None
    ):
        src_mask = _convert_attention_mask(attn_mask=attn_mask,
                                           dtype=hidden_states.dtype)

        output = hidden_states
        new_caches = []
        for i, mod in enumerate(self.layers):

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if cache is None:
                output = mod(output,
                             attn_mask=src_mask,
                             head_mask=layer_head_mask)
            else:
                # TODO can support cache?
                output, new_cache = mod(output,
                                        attn_mask=src_mask,
                                        head_mask=layer_head_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        return output if cache is None else (output, new_caches)


def get_extended_attention_mask(attention_mask, ):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask.unsqueeze(1)
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    else:
        raise ValueError(
            f"Wrong shape for attention_mask (shape {attention_mask.shape})"
        )
    extended_attention_mask = _convert_attention_mask(extended_attention_mask, "int64")
    return extended_attention_mask


# Copied from transformers/models/canine/modeling_canine.py _create_3d_attention_mask_from_input_mask
def _create_3d_attention_mask_from_input_mask(from_tensor, to_mask):
    """
    Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    batch_size, from_seq_length = from_tensor.shape[0], from_tensor.shape[1]

    to_seq_length = to_mask.shape[1]

    to_mask = paddle.reshape(to_mask, (batch_size, 1, to_seq_length))

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    broadcast_ones = paddle.ones(shape=(batch_size, from_seq_length, 1), dtype="int64")
    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


# Copied from transformers/models/canine/modeling_canine.py _downsample_attention_mask
def _downsample_attention_mask(char_attention_mask, downsampling_rate):
    """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

    # first, make char_attention_mask 3D by adding a channel dim
    batch_size, char_seq_len = char_attention_mask.shape
    poolable_char_mask = paddle.reshape(char_attention_mask, (batch_size, 1, char_seq_len))

    # next, apply MaxPool1d to get pooled_molecule_mask of shape (batch_size, 1, mol_seq_len)
    pooled_molecule_mask = nn.MaxPool1D(kernel_size=downsampling_rate,
                                        stride=downsampling_rate)(
        paddle.cast(poolable_char_mask, dtype_float)
    )
    # finally, squeeze to get tensor of shape (batch_size, mol_seq_len)
    molecule_attention_mask = paddle.squeeze(pooled_molecule_mask, axis=-1)
    return molecule_attention_mask


@register_base_model
class CanineModel(CaninePretrainedModel):
    """
    Construct a bare Canine Model.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Check the superclass documentation for the generic methods and the library implements for all its model.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        bos_token_id (`int`, optional):
            The id for begging of sentences token. Defaults to ``57344``.
        pad_token_id (`int`, optional):
            The id for padding token. Defaults to ``0``.
        eos_token_id (`int`, optional):
            The id for end of sentence token. Defaults to ``57345``.
        d_model (`int`, optional):
            Dimensionality of the layers and the pooler layer. Defaults to ``768``.
        num_encoder_layers (`int`, optional):
            Number of Transformer encoder layers for BlenderbotEncoder. Defaults to ``12``.
        num_heads (`int`, optional):
            Number of attention heads for each Transformer encoder layer in CanineEncoder.
            Defaults to ``12``.
        encoder_ffn_dim (`int`, optional):
            Dimensionality of the feed-forward layer for each Transformer encoder layer in
            CanineEncoder. Defaults to ``3072``.
        hidden_dropout (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            Defaults to ``0.1``.
        activation (`str`, optional):
            The non-linear activation function (function or string) in the encoder and pooler.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        attention_dropout (`float`, optional):
            The dropout ratio for the attention probabilities.
            Defaults to ``0.1``.
        max_position_embeddings (`int`, optional):,
            The max position index of an input sequence. Defaults to ``16384``.
        init_std (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
        layer_norm_eps(float, optional):
            The `epsilon` parameter used in :class:`paddle.nn.LayerNorm` for initializing layer normalization layers.
            A small value to the variance added to the normalization layer to prevent division by zero.
            Defaults to `1e-12`.
        type_vocab_size (int, optional):
            The vocabulary size of `token_type_ids`. Defaults to `16`.
        num_hash_functions (int, optional):
            The number of hash functions to construct hash embeddings. Defaults to `8`.
        num_hash_buckets (int, optional):
            The number of hash buckets for embedding hash function. Defaults to `16384`.
        downsampling_rate (int, optional):
            The down sampling rate for convert character sequence into molecules sequence. Defaults to `4`.
        upsampling_kernel_size (int, optional):
            The Convolution kernel size of projection layer after encoder. Defaults to `4`.
        add_pooling_layer (bool, optional):
            Whether to add pooling layer after encoder for classification task. Defaults to `True`.
        local_transformer_stride (int, optional):
            The stride size for performing single local attention in `CanineSelfAttention`. Default to `128`.
    """

    def __init__(self,
                 bos_token_id=57344,
                 eos_token_id=57345,
                 pad_token_id=0,
                 d_model=768,
                 num_encoder_layers=12,
                 num_heads=12,
                 hidden_dropout=0.01,
                 activation="gelu",
                 encoder_ffn_dim=3072,
                 attention_dropout=0.01,
                 max_position_embeddings=16384,
                 init_std=0.02,
                 layer_norm_eps=1e-12,
                 type_vocab_size=16,
                 num_hash_functions=8,
                 num_hash_buckets=16384,
                 downsampling_rate=4,
                 upsampling_kernel_size=4,
                 add_pooling_layer=True,
                 local_transformer_stride=128,
                 ):
        super(CanineModel, self).__init__()
        self.init_std = init_std
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_position_embeddings = max_position_embeddings
        self.downsampling_rate = downsampling_rate
        self.num_hidden_layers = num_encoder_layers
        # hash embedding
        self.char_embeddings = CanineEmbeddings(
            embedding_dim=d_model,
            num_hash_functions=num_hash_functions,
            num_hash_buckets=num_hash_buckets,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
        )
        # single local encoder
        self.initial_char_encoder = CanineEncoder(
            hidden_size=d_model,
            activation=activation,
            encoder_ffn_dim=encoder_ffn_dim,
            layer_norm_eps=layer_norm_eps,
            attn_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            num_heads=num_heads,
            num_encoder_layers=1,
            local=True,
            attend_from_chunk_width=local_transformer_stride,
            attend_from_chunk_stride=local_transformer_stride,
            attend_to_chunk_width=local_transformer_stride,
            attend_to_chunk_stride=local_transformer_stride,
        )

        # stride conv layer for down sampling
        self.chars_to_molecules = CharactersToMolecules(
            hidden_size=d_model,
            down_sampling_rate=downsampling_rate,
            activation=activation,
            layer_norm_eps=layer_norm_eps
        )

        # traditional transformer encoder
        self.encoder = CanineEncoder(
            hidden_size=d_model,
            activation=activation,
            encoder_ffn_dim=encoder_ffn_dim,
            layer_norm_eps=layer_norm_eps,
            attn_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
        )

        # Conv layer for up sampling
        self.projection = ConvProjection(
            hidden_size=d_model,
            upsampling_kernel_size=upsampling_kernel_size,
            activation=activation,
            hidden_dropout=hidden_dropout,
            layer_norm_eps=layer_norm_eps
        )
        # shallow/low-dim transformer encoder to get a final character encoding
        self.final_char_encoder = CanineEncoder(
            hidden_size=d_model,
            activation=activation,
            encoder_ffn_dim=encoder_ffn_dim,
            layer_norm_eps=layer_norm_eps,
            attn_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            num_heads=num_heads,
            num_encoder_layers=1
        )

        if add_pooling_layer:
            self.pooler = nn.Linear(d_model, d_model)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        self.apply(self.init_weights)

    # copy from `transformers.models.albert.modeling_albert.AlbertModel._convert_head_mask_to_5d`
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                -1)  # We can specify head_mask for each layer
        assert head_mask.dim(
        ) == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = paddle.cast(head_mask, dtype=dtype_float)
        return head_mask

    # copy from `transformers.models.albert.modeling_albert.AlbertModel.get_head_mask`
    def get_head_mask(self,
                      head_mask,
                      num_hidden_layers,
                      is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask,
                                                      num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _repeat_molecules(self, molecules, char_seq_length):
        """Repeats molecules to make them the same length as the char sequence."""

        rate = self.downsampling_rate

        molecules_without_extra_cls = molecules[:, 1:, :]
        # `repeated`: [batch_size, almost_char_seq_len, molecule_hidden_size]
        repeated = repeat_interleave(molecules_without_extra_cls, repeats=rate, axis=-2)
        # TODO check back paddle.repeat_interleave bug
        # So far, we've repeated the elements sufficient for any `char_seq_length`
        # that's a multiple of `downsampling_rate`. Now we account for the last
        # n elements (n < `downsampling_rate`), i.e. the remainder of floor
        # division. We do this by repeating the last molecule a few extra times.
        last_molecule = molecules[:, -1:, :]
        remainder_length = paddle.mod(paddle.to_tensor(char_seq_length), paddle.to_tensor(rate)).item()
        remainder_repeated = repeat_interleave(
            last_molecule,
            repeats=remainder_length + rate,
            axis=-2,
        )
        return paddle.concat([repeated, remainder_repeated], axis=-2)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,  # TODO check use or not
            inputs_embeds=None,
            return_dict=None,
    ):
        """
        The CanineModel forward method, overrides the `__call__()` special method.

        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            attention_mask (Tensor, optional):
                Mask to indicate whether to perform attention on each input token or not.
                The values should be either 0 or 1. The attention scores will be set
                to **-infinity** for any positions in the mask that are **0**, and will be
                **unchanged** for positions that are **1**.

                - **1** for tokens that are **not masked**,
                - **0** for tokens that are **masked**.

                It's data type should be `float32` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
            token_type_ids (Tensor, optional):
                 Segment token indices to indicate different portions of the inputs.
                 Selected in the range ``[0, type_vocab_size - 1]``.
                 If `type_vocab_size` is 2, which means the inputs have two portions.
                 Indices can either be 0 or 1:

                 - 0 corresponds to a *sentence A* token,
                 - 1 corresponds to a *sentence B* token.

                 Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                 Defaults to `None`, which means we don't add segment embeddings.


            position_ids(Tensor, optional):
                 Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
                 max_position_embeddings - 1]``.
                 Shape as `(batch_size, num_tokens)` and dtype as int64. Defaults to `None`.
            head_mask (Tensor, optional):
                 Mask to nullify selected heads of the self-attention modules. Masks values can either be 0 or 1:

                 - 1 indicates the head is **not masked**,
                 - 0 indicated the head is **masked**.
            inputs_embeds (Tensor, optional):
                If you want to control how to convert `inputs_ids` indices into associated vectors, you can
                pass an embedded representation directly instead of passing `inputs_ids`.
            return_dict (bool, optional):
                Whether to return a dict instead of a plain tuple. Default to `False`.

        Returns:
            tuple or Dict: Returns tuple (`sequence_output`, `pooled_output`) or a dict with
             `last_hidden_state`, `pooled_output`
            With the fields:

            - `sequence_output` (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and has a shape of [`batch_size, sequence_length, hidden_size`].

            - `pooled_output` (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and
                has a shape of [batch_size, hidden_size].
        Example:
            .. code-block::

            from paddlenlp.transformers import CanineTokenizer
            from paddlenlp.transformers import CanineModel
            import paddle
            tokenizer = CanineTokenizer.from_pretrained('canine-s')
            model = CanineModel.from_pretrained('canine-s')

            text = ["Canine model is tokenization-free-free."]
            inputs = tokenizer(text)
            inputs = {k:paddle.to_tensor(v) for (k, v) in inputs.items()}

            output = model(**inputs)

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(shape=input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(shape=input_shape, dtype="int64")

        extended_attention_mask = get_extended_attention_mask(attention_mask)
        molecule_attention_mask = _downsample_attention_mask(attention_mask,
                                                             downsampling_rate=self.downsampling_rate
                                                             )
        extended_molecule_attention_mask = get_extended_attention_mask(molecule_attention_mask)
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)
        # TODO check head mask is not used

        # `input_char_embeddings`: shape (batch_size, char_seq, char_dim)
        input_char_embeddings = self.char_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        char_attention_mask = _create_3d_attention_mask_from_input_mask(input_ids, attention_mask)

        # single local transformer for encoding character sequence.
        # `input_char_encoding`: shape (batch_size, char_seq, hidden_size)
        input_char_encoding = self.initial_char_encoder(
            input_char_embeddings,
            attn_mask=char_attention_mask,
        )

        # Downsample chars to molecules.
        # The following lines have dimensions: [batch, molecule_seq_len, molecule_dim].
        # In this transformation, we change the dimensionality from `char_dim` to
        # `molecule_dim`, but do *NOT* add a resnet connection. Instead, we rely on
        # the resnet connections (a) from the final char transformer stack back into
        # the original char transformer stack and (b) the resnet connections from
        # the final char transformer stack back into the deep BERT stack of
        # molecules.
        #
        # Empirically, it is critical to use a powerful enough transformation here:
        # mean pooling causes training to diverge with huge gradient norms in this
        # region of the model; using a convolution here resolves this issue. From
        # this, it seems that molecules and characters require a very different
        # feature space; intuitively, this makes sense.

        # Refer to the Paper, `molecule_seq_len == char_seq_len/downsampling_rate`
        # , and `molecule_dim` is just same as hidden size
        init_molecule_encoding = self.chars_to_molecules(input_char_encoding)

        # Traditional Transformer Encoder
        # `molecule_sequence_output`: shape (batch_size, mol_seq_len, molecule_dim)
        molecule_sequence_output = self.encoder(init_molecule_encoding,
                                                attn_mask=extended_molecule_attention_mask,
                                                head_mask=head_mask,
                                                )

        pooled_output = self.pooler_activation(self.pooler(molecule_sequence_output[:, 0])) \
            if self.pooler is not None else None

        # `repeated_molecules`: shape (batch_size, char_seq_len, hidden_size)
        repeated_molecules = self._repeat_molecules(molecule_sequence_output,
                                                    char_seq_length=input_shape[-1])
        # `concat`: shape (batch_size, char_seq_len, 2*hidden_size)
        concat = paddle.concat([input_char_encoding, repeated_molecules], axis=-1)

        # `sequence_output`: shape (batch_size, char_seq_len, hidden_size)
        sequence_output = self.projection(concat)

        # final transformer layer
        sequence_output = self.final_char_encoder(
            sequence_output,
            attn_mask=extended_attention_mask,
        )
        if return_dict:
            return {
                "last_hidden_state": sequence_output,
                "pooler_output": pooled_output,
            }
        return sequence_output, pooled_output


def repeat_interleave(input, repeats, **kwargs):
    # repeat interleave for canine model only!! (avoid paddle.interleave .backward() bug)
    # it will be removed when paddle.interleave fixed.
    batch_size, len_seq, d_model = input.shape
    flat_input = paddle.tile(input, repeat_times=(1, 1, repeats))
    return paddle.reshape(flat_input, [batch_size, -1, d_model])
