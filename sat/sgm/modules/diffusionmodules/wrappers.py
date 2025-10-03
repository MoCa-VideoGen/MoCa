import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, x_reference: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        for key in c:
            c[key] = c[key].to(self.dtype)
        if x.dim() == 4:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        elif x.dim() == 5:
            x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=2)
            x_reference = torch.cat((x_reference, c.get("concat", torch.Tensor([]).type_as(x_reference))), dim=2)
        else:
            raise ValueError("Input tensor must be 4D or 5D")

        return self.diffusion_model(#DiffusionTransformer(  (mixins): ModuleDict(    (pos_embed): Rotary3DPositionEmbeddingMixin()    (patch_embed): ImagePatchEmbeddingMixin(      (proj): Conv2d(16, 3072, kernel_size=(2, 2), stride=(2, 2))      (text_proj): Linear(in_features=4096, out_features=3072, bias=True)    )    (adaln_layer): AdaLNMixin(      (adaLN_modulations): ModuleList(        (0-41): 42 x Sequential(          (0): SiLU()          (1): Linear(in_features=512, out_features=36864, bias=True)        )      )      (query_layernorm_list): ModuleList(        (0-41): 42 x LayerNorm((64,), eps=1e-06, elementwise_affine=True)      )      (key_layernorm_list): ModuleList(        (0-41): 42 x LayerNorm((64,), eps=1e-06, elementwise_affine=True)      )      (fuser_list): ModuleList(        (0-41): 42 x MGF(          (flow_gamma_spatial): Conv2d(128, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))          (flow_gamma_temporal): Conv1d(768, 3072, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=replicate)          (flow_beta_spatial): Conv2d(128, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))          (flow_beta_temporal): Conv1d(768, 3072, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=replicate)          (flow_cond_norm): FloatGroupNorm(32, 3072, eps=1e-05, affine=True)        )      )    )    (final_layer): FinalLayerMixin(      (norm_final): LayerNorm((3072,), eps=1e-06, elementwise_affine=True)      (linear): Linear(in_features=3072, out_features=64, bias=True)      (adaLN_modulation): Sequential(        (0): SiLU()        (1): Linear(in_features=512, out_features=6144, bias=True)      )    )  )  (transformer): BaseTransformer(    (embedding_dropout): Dropout(p=0, inplace=False)    (position_embeddings): Embedding(64, 3072)    (layers): ModuleList(      (0-41): 42 x BaseTransformerLayer(        (input_layernorm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)        (attention): SelfAttention(          (query_key_value): ColumnParallelLinear()          (attention_dropout): Dropout(p=0, inplace=False)          (dense): RowParallelLinear()          (output_dropout): Dropout(p=0, inplace=False)        )        (post_attention_layernorm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)        (mlp): MLP(          (activation_func): GELU(approximate='tanh')          (dense_h_to_4h): ColumnParallelLinear()          (dense_4h_to_h): RowParallelLinear()          (dropout): Dropout(p=0, inplace=False)        )      )    )    (final_layernorm): LayerNorm((3072,), eps=1e-05, elementwise_affine=True)  )  (time_embed): Sequential(    (0): Linear(in_features=3072, out_features=512, bias=True)    (1): SiLU()    (2): Linear(in_features=512, out_features=512, bias=True)  )  (traj_extractor): TrajExtractor(    (downsize_patchify): PixelUnshuffle(downscale_factor=2)    (body): ModuleList(      (0-83): 84 x ResnetBlock(        (block1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))        (act): ReLU()        (block2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)      )    )    (conv_in): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  ))
            x,
            x_reference,
            timesteps=t,
            context=c.get("crossattn", None),#tensor([[[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],         ...,         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]],        [[-0.2363,  0.1934,  0.0388,  ..., -0.1865, -0.0466, -0.0972],         [-0.3008,  0.1182,  0.0216,  ...,  0.0035, -0.1123,  0.0762],         [ 0.0588,  0.1416,  0.0315,  ...,  0.0593, -0.0530,  0.1582],         ...,         [ 0.0596, -0.0140,  0.0479,  ..., -0.0032, -0.0486, -0.0220],         [-0.0320,  0.0591, -0.0325,  ..., -0.2393, -0.1157, -0.0669],         [ 0.0049,  0.0206, -0.0141,  ..., -0.0057, -0.0135,  0.0054]]],       device='cuda:0', dtype=torch.bfloat16)
            y=c.get("vector", None), #None
            **kwargs,
        )
        #sat/dit_video_concat.py P1114
