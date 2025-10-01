# coding=utf-8
# -*- encoding: utf-8 -*-
'''
@File    :   transformer_defaults.py
@Time    :   2022/06/01 21:44:17
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

import math
import torch
import torch.nn.functional as F

from sat import mpu

from sat.mpu.utils import split_tensor_along_last_dim
import contextlib


def save_text_to_vis_attn(attn_probs, text_length, token_idx, avg_heads=True):
    """
    保存指定 text token 对 visual token 的 attention map
    attn_probs: [B, num_heads, L_query, L_key]
    text_length: 文本 token 总数
    token_idx: 要查看的文本 token index (0 <= token_idx < text_length)
    avg_heads: 是否对多个 head 求平均
    """
    global _saved_attn_map
    # 取 batch=0，这里也可以改成循环取多个 batch
    vis_map = attn_probs[0, :, token_idx, text_length:]  # [num_heads, n_image_tokens]
    if avg_heads:
        vis_map = vis_map.mean(dim=0)  # [n_image_tokens]
    return vis_map

def scaled_dot_product_with_save(query, key, value, attn_mask=None,
                               dropout_p=0.0, is_causal=False,
                               save_cfg=None):
    T, H, W = 13, 30, 45  # 可视化尺寸
    # q k v : [2,48,17776,64]
    # 主路径：高效计算注意力输出
    attn_output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal
    )

    # 可视化路径：按需计算
    vis_attn = None
    if save_cfg is not None:
        text_length = save_cfg["text_length"]
        token_idx = save_cfg["token_idx"]
        # query:[2,48,17776,64]
        # 计算目标token的注意力（已经是 [B, heads, 1, K]） [B,48,1,64] key:[B,48,17776,64]
        target_query = query[:, :, token_idx:token_idx+1, :] #[2,48,1,64]
        attn_scores = torch.matmul(target_query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask[:, :, token_idx:token_idx+1, :]
        attn_probs = torch.softmax(attn_scores, dim=-1)  # [2, 48, 1, 17776]
        
        # 直接提取visual部分，无需再次索引token_idx [B,48,1,17776]
        vis_map = attn_probs[1, :, 0, text_length:]  # [heads, n_image_tokens]
        if save_cfg.get("avg_heads", True):
            vis_map = vis_map.mean(dim=0)  # [n_image_tokens] [17550]
        vis_attn = vis_map.view(T, H, W)

    return vis_attn, attn_output

def standard_attention(query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    if log_attention_weights is not None:
        attention_scores += log_attention_weights

    if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
        # if auto-regressive, skip
        attention_scores = torch.mul(attention_scores, attention_mask) - \
                           10000.0 * (1.0 - attention_mask)

    attention_probs = F.softmax(attention_scores, dim=-1)

    if attention_dropout is not None:
        if mpu.get_cuda_rng_tracker is not None:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = attention_dropout(attention_probs)
        else:
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def attention_fn_default(query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    # expand head dim to query dim, if necessary
    # only useful for multi-query attention
    batch_size, num_query_heads = query_layer.shape[:2] # [b, np, s, hn]
    num_kv_heads = key_layer.shape[1] #  48
    key_layer = key_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *key_layer.shape[2:])
    value_layer = value_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *value_layer.shape[2:])
    #key_layer [B,48,17776,64]   value_layer [2,48,17776,64]
    is_low_triangle = (attention_mask == torch.ones_like(attention_mask, dtype=torch.float).tril()).all()
    is_full = (attention_mask is None) or (attention_mask > 0).all()

    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        # Pytorch 2.0 attention uses very much memory if attention_mask is float, and has NaN bug if attention_mask is None.
        dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        if dropout_p > 0 and mpu.get_cuda_rng_tracker is not None:
            context = mpu.get_cuda_rng_tracker().fork()
        else:
            context = contextlib.nullcontext()
        with context:
            save_cfg = kwargs['save_cfg'] 
            vis_attn,attn_output =  scaled_dot_product_with_save(query_layer, key_layer, value_layer, attn_mask=None,
                               dropout_p=dropout_p, is_causal=not is_full, save_cfg = save_cfg
                              )
            #attn_output = torch.nn.functional.scaled_dot_product_attention(
            #    query_layer, key_layer, value_layer, 
            #    attn_mask=None,
            #    dropout_p=dropout_p,
            #    is_causal=not is_full
            #)
        return  vis_attn , attn_output
    else:
        return standard_attention(
            query_layer, key_layer, value_layer, attention_mask,
            attention_dropout=attention_dropout, log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score, **kwargs
        )

def attention_forward_default(self, hidden_states, mask, **kw_args): #hidden_states[B, 17550+226, Channel]
    self = self.transformer.layers[kw_args['layer_id']].attention
    attention_fn = attention_fn_default
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_raw_layer = self.query_key_value(hidden_states)
    (mixed_query_layer,
        mixed_key_layer,
        mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, self.stride)

    dropout_fn = self.attention_dropout if self.training else None

    query_layer = self._transpose_for_scores(mixed_query_layer) #[2,48,17776,64]
    key_layer = self._transpose_for_scores(mixed_key_layer) #[2,48,17776,64]
    value_layer = self._transpose_for_scores(mixed_value_layer)  #[2,48,17776,64]

    # rotary position embedding 
    if self.transformer.is_rotary_emb:
        query_layer, key_layer = self.transformer.position_embeddings(
            query_layer, key_layer, kw_args['position_ids'],max_seqlen=kw_args['position_ids'].max()+1,
            layer_id=kw_args['layer_id']
        )

    vis_attn , context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)
    #dit.py/P315
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)
    output = self.dense(context_layer)

    if self.training:
        output = self.output_dropout(output)
    return vis_attn , output

def cross_attention_forward_default(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].cross_attention
    attention_fn = attention_fn_default
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_query_layer = self.query(hidden_states)
    query_layer = self._transpose_for_scores(mixed_query_layer)
    dropout_fn = self.attention_dropout if self.training else None
    if isinstance(encoder_outputs, torch.Tensor):
        mixed_x_layer = self.key_value(encoder_outputs)
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)
        # Reshape and transpose [b, np, s, hn]
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        mem_cross = (key_layer, value_layer)
    else:
        key_layer, value_layer = encoder_outputs[kw_args['layer_id']]
        mem_cross = (key_layer, value_layer)

    context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn, cross_attention=True, mem_cross=mem_cross, **kw_args)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    # [b, s, hp]
    context_layer = context_layer.view(*new_context_layer_shape)

    # Output. [b, s, h]
    output = self.dense(context_layer)
    if self.training:
        output = self.output_dropout(output)
    return output

def routing_forward_default(self, hidden_states, **kw_args):
    num_experts = self.transformer.num_experts
    # This is just an example that select 2 experts randomly.
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = torch.randn((batch_size*sequence_length, num_experts), device=hidden_states.device, dtype=hidden_states.dtype)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, 2, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)
    return routing_weights, selected_experts

from functools import partial

def mlp_forward_default(self, hidden_states, expert_id=-1, **kw_args):
    if self.transformer.num_experts == 1 or expert_id > -1:
        self = self.transformer.layers[kw_args['layer_id']].mlp
        suffix = f"_{expert_id}" if expert_id > 0 else ""
        if self.is_gated_mlp:
            intermediate_parallel = getattr(self, "dense_h_to_4h"+suffix)(hidden_states)
            gated_intermediate_parallel = getattr(self, "dense_h_to_4h_gate"+suffix)(hidden_states)
            intermediate_parallel = self.activation_func(gated_intermediate_parallel) * intermediate_parallel
            output = getattr(self, "dense_4h_to_h"+suffix)(intermediate_parallel)
        else:
            intermediate_parallel = getattr(self, "dense_h_to_4h"+suffix)(hidden_states)
            intermediate_parallel = self.activation_func(intermediate_parallel)
            output = getattr(self, "dense_4h_to_h"+suffix)(intermediate_parallel)
        return output
    else:
        mlp_forward = self.hooks.get('mlp_forward', partial(mlp_forward_default, self))
        routing_forward = self.hooks.get('routing_forward', partial(routing_forward_default, self))
        self = self.transformer.layers[kw_args['layer_id']].mlp
        fwd_weight, fwd_idx = routing_forward(hidden_states, **kw_args)

        # Adapted from mixtral-8x7b https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(fwd_idx, num_classes=self.num_experts).permute(2, 1, 0)
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue
            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[top_x_list] # I don't know why using hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = mlp_forward(current_state, expert_id=expert_idx, **kw_args) * fwd_weight[top_x_list, idx_list, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        output = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return output

def word_embedding_forward_default(self, input_ids, output_cross_layer, **kw_args):
    return self.transformer.word_embeddings(input_ids)

def position_embedding_forward_default(self, position_ids, output_cross_layer, **kw_args):
    if not self.transformer.is_rotary_emb:
        return self.transformer.position_embeddings(position_ids)
    return None
        
from sat.mpu import gather_from_model_parallel_region
def final_forward_default(self, logits, **kw_args):
    logits_parallel = F.linear(logits, self.transformer.word_embeddings.weight)
    if not kw_args['parallel_output']:
        logits_parallel = gather_from_model_parallel_region(logits_parallel)
    return logits_parallel

def layer_forward_default(self, hidden_states, mask, *args, **kw_args):
    '''
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
    '''
    self = self.transformer.layers[kw_args['layer_id']]
    
    # Layer norm at the begining of the transformer layer.
    attention_input = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output = self.attention(attention_input, mask, **kw_args)

    # Third LayerNorm
    if self.layernorm_order == 'sandwich':
        attention_output = self.third_layernorm(attention_output)

    # DropPath for attention
    if self.training and self.drop_path > 0.:
        # drop_path percentage 0, others 1/(1-p)
        random_tensor = (1-self.drop_path
                            + torch.rand((attention_output.shape[0],), dtype=attention_output.dtype, device=attention_output.device)).floor_() / (1-self.drop_path)
        attention_output = random_tensor.view(-1, 1, 1) * attention_output
    
    # Residual connection.
    if self.layernorm_order == 'post':
        hidden_states = attention_input + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)
    else:
        hidden_states = hidden_states + attention_output

    if self.is_decoder:
        encoder_outputs = kw_args['encoder_outputs']
        if encoder_outputs is not None:
            assert 'cross_attention_mask' in kw_args
            # Cross attention
            if self.layernorm_order == 'post':
                attention_output = self.cross_attention(mlp_input, **kw_args)
                # Residual connection.
                hidden_states = mlp_input + attention_output
                # Layer norm post the cross attention
                mlp_input = self.post_cross_attention_layernorm(hidden_states)
            else:
                cross_input = self.post_cross_attention_layernorm(hidden_states)
                attention_output = self.cross_attention(cross_input, **kw_args)
                hidden_states = hidden_states + attention_output

    if self.layernorm_order != 'post':
        mlp_input = self.post_attention_layernorm(hidden_states)    

    # MLP.
    mlp_output = self.mlp(mlp_input, **kw_args)

    # Fourth LayerNorm
    if self.layernorm_order == 'sandwich':
        mlp_output = self.fourth_layernorm(mlp_output)

    # DropPath for mlp
    if self.training and self.drop_path > 0.:
        random_tensor = (1-self.drop_path
                            + torch.rand((mlp_output.shape[0],), dtype=mlp_output.dtype, device=mlp_output.device)).floor_() / (1-self.drop_path)
        mlp_output = random_tensor.view(-1, 1, 1) * mlp_output

    # Second residual connection.
    if self.layernorm_order == 'post':
        output = mlp_input + mlp_output
    else:
        output = hidden_states + mlp_output

    return output

HOOKS_DEFAULT = {
    'attention_fn': attention_fn_default,
    'attention_forward': attention_forward_default,
    'cross_attention_forward': cross_attention_forward_default,
    'routing_forward': routing_forward_default,
    'mlp_forward': mlp_forward_default,
    'word_embedding_forward': word_embedding_forward_default,
    'position_embedding_forward': position_embedding_forward_default,
    'final_forward': final_forward_default,
    'layer_forward': layer_forward_default
}

ARGS_DEFAULT = {
    'embedding_dropout_prob': ('hidden_dropout', 0),
    'attention_dropout_prob': ('attention_dropout', 0),
    'output_dropout_prob': ('hidden_dropout', 0),
    'inner_hidden_size': ('inner_hidden_size', None),
    'hidden_size_per_attention_head': ('hidden_size_per_attention_head', None),
    'cross_hidden_size_per_attention_head': ('cross_hidden_size_per_attention_head', None),
    'checkpoint_activations': ('checkpoint_activations', False),
    'checkpoint_num_layers': ('checkpoint_num_layers', 1),
    'checkpoint_skip_layers': ('checkpoint_skip_layers', 0),
    'is_decoder': ('is_decoder', False),
    'cross_attn_hidden_size': ('cross_attn_hidden_size', None),
    'use_final_layernorm': ('use_final_layernorm', True),
    'layernorm_epsilon': ('layernorm_epsilon', 1e-5),
    'use_bias': ('use_bias', True),
    'use_qkv_bias': ('use_qkv_bias', False),
    'num_multi_query_heads': ('num_multi_query_heads', 0),
    'cross_num_multi_query_heads': ('cross_num_multi_query_heads', 0),
    'drop_path': ('drop_path', 0.),
    'row_parallel_linear_final_bias': ('row_parallel_linear_final_bias', True),
    'is_gated_mlp': ('is_gated_mlp', False),
    'is_rotary_emb': ('is_rotary_emb', False),
    'parallel_output': ('parallel_output', False),
    'num_experts': ('num_experts', 1),
}

from sat.ops.layernorm import LayerNorm, RMSNorm

NO_WD_MODULES = [LayerNorm, torch.nn.LayerNorm, RMSNorm]