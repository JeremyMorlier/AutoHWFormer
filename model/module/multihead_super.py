import torch
from torch import nn
import torch.nn.functional as F
from .Linear_super import LinearSuper
from .qkv_super import qkv_super
from ..utils import trunc_normal_

from torch.profiler import record_function


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class RelativePosition2D_super(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding for the class
        self.embeddings_table_v = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units)
        )
        self.embeddings_table_h = nn.Parameter(
            torch.randn(max_relative_position * 2 + 2, num_units)
        )

        trunc_normal_(self.embeddings_table_v, std=0.02)
        trunc_normal_(self.embeddings_table_h, std=0.02)

        self.sample_head_dim = None
        self.sample_embeddings_table_h = None
        self.sample_embeddings_table_v = None

    def set_sample_config(self, sample_head_dim):
        self.sample_head_dim = sample_head_dim
        self.sample_embeddings_table_h = self.embeddings_table_h[:, :sample_head_dim]
        self.sample_embeddings_table_v = self.embeddings_table_v[:, :sample_head_dim]

    def calc_sampled_param_num(self):
        return (
            self.sample_embeddings_table_h.numel()
            + self.sample_embeddings_table_v.numel()
        )

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        device = self.embeddings_table_v.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        # compute the row and column distance
        distance_mat_v = range_vec_k[None, :] // int(length_q**0.5) - range_vec_q[
            :, None
        ] // int(length_q**0.5)
        distance_mat_h = range_vec_k[None, :] % int(length_q**0.5) - range_vec_q[
            :, None
        ] % int(length_q**0.5)
        # clip the distance to the range of [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(
            distance_mat_v, -self.max_relative_position, self.max_relative_position
        )
        distance_mat_clipped_h = torch.clamp(
            distance_mat_h, -self.max_relative_position, self.max_relative_position
        )

        # translate the distance from [1, 2 * max_relative_position + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1, 0, 1, 0), "constant", 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1, 0, 1, 0), "constant", 0)

        final_mat_v = final_mat_v.long().detach()
        final_mat_h = final_mat_h.long().detach()
        # get the embeddings with the corresponding distance
        embeddings = (
            self.sample_embeddings_table_v[final_mat_v]
            + self.sample_embeddings_table_h[final_mat_h]
        )

        return embeddings


class AttentionSuper(nn.Module):
    def __init__(
        self,
        super_embed_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        normalization=False,
        relative_position=False,
        num_patches=None,
        max_relative_position=14,
        scale=False,
        change_qkv=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.super_embed_dim = super_embed_dim

        self.fc_scale = scale
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)
        else:
            self.qkv = LinearSuper(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position
            )
            self.rel_pos_embed_v = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position
            )
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None

        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(
        self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None
    ):
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (sample_in_embed_dim // self.sample_num_heads) ** -0.5

        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (
                self.sample_qk_embed_dim // self.sample_num_heads
            ) ** -0.5

        self.qkv.set_sample_config(
            sample_in_dim=sample_in_embed_dim,
            sample_out_dim=3 * self.sample_qk_embed_dim,
        )
        self.proj.set_sample_config(
            sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim
        )
        if self.relative_position:
            self.rel_pos_embed_k.set_sample_config(
                self.sample_qk_embed_dim // sample_num_heads
            )
            self.rel_pos_embed_v.set_sample_config(
                self.sample_qk_embed_dim // sample_num_heads
            )

    def calc_sampled_param_num(self):
        return 0

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.qkv.get_complexity(sequence_length)
        # attn
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        # x
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        total_flops += self.proj.get_complexity(sequence_length)
        if self.relative_position:
            total_flops += (
                self.max_relative_position * sequence_length * sequence_length
                + sequence_length * sequence_length / 2.0
            )
            total_flops += (
                self.max_relative_position * sequence_length * sequence_length
                + sequence_length * self.sample_qk_embed_dim / 2.0
            )
        return total_flops

    def forward(self, x):
        with record_function("attn"):
            B, N, C = x.shape
            # print(x.shape, B, N, 3, self.sample_num_heads, -1)
            # qkv = self.qkv(x)
            # print(qkv.shape)
            # qkv = (
            #     qkv
            #     .reshape(B, N, 3, self.sample_num_heads, -1)
            #     .permute(2, 0, 3, 1, 4)
            # )
            # q, k, v = (
            #     qkv[0],
            #     qkv[1],
            #     qkv[2],
            # )  # make torchscript happy (cannot use tensor as tuple)

            qkv = self.qkv(x).reshape(B, N, 3, self.sample_num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            if self.relative_position:
                print(q.shape, self.rel_pos_embed_k(N, N).shape)
                q = q + self.rel_pos_embed_k(N, N).unsqueeze(0)
                v = v + self.rel_pos_embed_v(N, N).unsqueeze(0)

            # Use PyTorch's built-in scaled dot product attention
            attn_output, attn_weights = F.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                dropout_p=self.dropout if self.training else 0.0
            )

            x = attn_output.transpose(0, 1).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x
