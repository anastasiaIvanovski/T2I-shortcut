# Project part of GenerativeAI course
# --------------------------------------------------------
# References:
# Shortcut model: https://github.com/kvfrans/shortcut-models
# PyTorch Version: https://github.com/smileyenot983/shortcut_pytorch
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from os import makedirs
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
from utils import create_targets, create_targets_naive
from typing import Dict, Any
from typing import Iterable, Optional
import weakref
import copy
import contextlib
from transformers import CLIPTokenizer, CLIPTextModel


LEARNING_RATE = 0.0001
EVAL_SIZE = 8
NUM_CLASSES = 1
CFG_SCALE = 0.0

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def to_float_maybe(x):
    return x.float() if x.dtype in [torch.float16, torch.bfloat16] else x


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    Args:
        parameters: Iterable of `torch.nn.Parameter` (typically from
            `model.parameters()`).
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing
            averages.
    """
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float,
        use_num_updates: bool = True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        parameters = list(parameters)
        self.shadow_params = [to_float_maybe(p.clone().detach())
                              for p in parameters if p.requires_grad]
        self.collected_params = None
        self._params_refs = [weakref.ref(p) for p in parameters]

    def _get_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]]
    ) -> Iterable[torch.nn.Parameter]:
        if parameters is None:
            parameters = [p() for p in self._params_refs]
            if any(p is None for p in parameters):
                raise ValueError(
                    "(One of) the parameters with which this "
                    "ExponentialMovingAverage "
                    "was initialized no longer exists (was garbage collected);"
                    " please either provide `parameters` explicitly or keep "
                    "the model to which they belong from being garbage "
                    "collected."
                )
            return parameters
        else:
            parameters = list(parameters)
            if len(parameters) != len(self.shadow_params):
                raise ValueError(
                    "Number of parameters passed as argument is different "
                    "from number of shadow parameters maintained by this "
                    "ExponentialMovingAverage"
                )
            return parameters

    def update(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(
                decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )
        one_minus_decay = 1.0 - decay
        if parameters[0].device != self.shadow_params[0].device:
            self.to(device=parameters[0].device)
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                torch.lerp(s_param, param.to(dtype=s_param.dtype), one_minus_decay, out=s_param)

    def copy_to(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Save the current parameters for restoring later.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored. If `None`, the parameters of with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.collected_params = [
            param.clone()
            for param in parameters
            if param.requires_grad
        ]

    def restore(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ) -> None:
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        if self.collected_params is None:
            raise RuntimeError(
                "This ExponentialMovingAverage has no `store()`ed weights "
                "to `restore()`"
            )
        parameters = self._get_parameters(parameters)
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)

    @contextlib.contextmanager
    def average_parameters(
        self,
        parameters: Optional[Iterable[torch.nn.Parameter]] = None
    ):
        r"""
        Context manager for validation/inference with averaged parameters.
        Equivalent to:
            ema.store()
            ema.copy_to()
            try:
                ...
            finally:
                ema.restore()
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = self._get_parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype)
            if p.is_floating_point()
            else p.to(device=device)
            for p in self.shadow_params
        ]
        if self.collected_params is not None:
            self.collected_params = [
                p.to(device=device, dtype=dtype)
                if p.is_floating_point()
                else p.to(device=device)
                for p in self.collected_params
            ]
        return

    def state_dict(self) -> dict:
        r"""Returns the state of the ExponentialMovingAverage as a dict."""
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""Loads the ExponentialMovingAverage state.
        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.num_updates = state_dict["num_updates"]
        assert self.num_updates is None or isinstance(self.num_updates, int), \
            "Invalid num_updates"

        self.shadow_params = state_dict["shadow_params"]
        assert isinstance(self.shadow_params, list), \
            "shadow_params must be a list"
        assert all(
            isinstance(p, torch.Tensor) for p in self.shadow_params
        ), "shadow_params must all be Tensors"

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            assert isinstance(self.collected_params, list), \
                "collected_params must be a list"
            assert all(
                isinstance(p, torch.Tensor) for p in self.collected_params
            ), "collected_params must all be Tensors"
            assert len(self.collected_params) == len(self.shadow_params), \
                "collected_params and shadow_params had different lengths"

        if len(self.shadow_params) == len(self._params_refs):
            # Consistent with torch.optim.Optimizer, cast things to consistent
            # device and dtype with the parameters
            params = [p() for p in self._params_refs]
            # If parameters have been garbage collected, just load the state
            # we were given without change.
            if not any(p is None for p in params):
                # ^ parameter references are still good
                for i, p in enumerate(params):
                    self.shadow_params[i] = to_float_maybe(self.shadow_params[i].to(
                        device=p.device, dtype=p.dtype
                    ))
                    if self.collected_params is not None:
                        self.collected_params[i] = self.collected_params[i].to(
                            device=p.device, dtype=p.dtype
                        )
        else:
            raise ValueError(
                "Tried to `load_state_dict()` with the wrong number of "
                "parameters in the saved state."
            )


class EMACallback(pl.Callback):
    def __init__(self, decay: float, use_num_updates: bool = True):
        """
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing averages.
        """
        super().__init__()
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.ema = None

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # If already loaded EMA from the checkpoint
        if self.ema is None:
          self.ema = ExponentialMovingAverage([p for p in pl_module.parameters() if p.requires_grad],
                                              decay=self.decay, use_num_updates=self.use_num_updates)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
          self.ema.update()

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # During the initial validation we don't have self.ema yet
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema is not None:
            self.ema.restore()

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.ema is not None:
            self.ema.restore()

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self.ema.state_dict()

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any]
    ) -> None:
        if self.ema is None:
            self.ema = ExponentialMovingAverage([p for p in pl_module.parameters() if p.requires_grad],
                                                decay=self.decay, use_num_updates=self.use_num_updates)
        self.ema.load_state_dict(checkpoint)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(pl.LightningModule):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)

        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TextEmbedder(pl.LightningModule):
    """
    Embeds text into vector representations.
    """
    def __init__(self, embedding_dim=768, eval_mode=True):
        super(TextEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        self.text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to('cuda')

        # Set the model to evaluation mode if needed
        if eval_mode:
            self.text_model.eval()

        # Projection layer to change the embedding size from 512 to 768
        self.projection_layer = torch.nn.Linear(512, embedding_dim).to('cuda')

    def forward(self, text, truncation=True, padding='max_length', max_length=77):
        """
        Forward pass to obtain text embeddings.
        - text: List of input texts (batch_size,)
        - truncation: Whether to truncate input to max_length.
        - padding: Padding strategy.
        - max_length: Maximum length of the input sequence.
        """
        # Tokenize the input text
        token_output = self.text_tokenizer(text,
                                           truncation=truncation,
                                           padding=padding,
                                           return_attention_mask=True,
                                           max_length=max_length)
        indexed_tokens = token_output['input_ids']
        att_masks = token_output['attention_mask']

        # Convert tokens to tensors and move to device
        tokens_tensor = torch.tensor(indexed_tokens).to('cuda')
        mask_tensor = torch.tensor(att_masks).to('cuda')

        # Get the embeddings from CLIP (shape: batch_size, seq_len, hidden_size)
        text_embed = self.text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state
        # Project from 512 to the desired embedding size (768)
        text_embed = self.projection_layer(text_embed)  # (batch_size, 77, 768)
        # Apply mean pooling across the sequence dimension (77 tokens)
        text_embed = text_embed.mean(dim=1)  # Shape: (batch_size, hidden_size)
        return text_embed


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(pl.LightningModule):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(pl.LightningModule):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(pl.LightningModule):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        training_type="shortcut"
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.dt_embedder = TimestepEmbedder(hidden_size)
        self.text_embedder = TextEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()


        self.training_type = training_type

        # number of denoising steps to be applied
        self.denoise_timesteps = [1, 2, 4, 8, 16, 32, 128]

        self.fids = None
        self.validation_step_outputs = []
        
        makedirs("log_images3", exist_ok=True)
        

    def on_fit_start(self):

        self.fids = [FrechetInceptionDistance().to(self.device) for _ in range(len(self.denoise_timesteps))]


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.text_embedder.projection_layer.weight, std=0.02)
        nn.init.constant_(self.text_embedder.projection_layer.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.dt_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.dt_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        self.loss_fn = torch.nn.MSELoss()

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, dt, text):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        if self.training_type=="naive":
            dt = torch.zeros_like(t)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        dt = self.dt_embedder(dt)
        text = self.text_embedder(text)    # (N, D)
        # print(f"t.shape: {t.shape}")
        # print(f"y.shape: {y.shape}")
        # print(f"dt.shape: {dt.shape}")
        # print(f"x.shape: {x.shape}")
        c = t + text + dt                              # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x
    
    def training_step(self, batch, batch_idx):

        images, text = batch

        text = torch.ones_like(text)

        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        if self.training_type=="naive":
            x_t, v_t, t, dt_base, _ = create_targets_naive(latents, text, self)
        elif self.training_type=="shortcut":
            x_t, v_t, t, dt_base, _ = create_targets(latents, text, self)

        v_prime = self.forward(x_t, t, dt_base, text)

        loss = self.loss_fn(v_prime, v_t)
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        images, text_real = batch


        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # normalize to [0,255] range
        images = 255 * ((images - torch.min(images)) / (torch.max(images) - torch.min(images) + 1e-8))

        # sample noise
        eps_i = torch.randn_like(latents).to(self.device)

        # denoise_timesteps_list = [1, 2, 4, 8, 16, 32, 128]

        for i, denoise_timesteps in enumerate(self.denoise_timesteps):

            all_x = []
            delta_t = 1.0 / denoise_timesteps # i.e. step size

            x = eps_i.to(self.device)

            for ti in range(denoise_timesteps):
                # t should in range [0,1]
                t = ti / denoise_timesteps

                t_vector = torch.full((eps_i.shape[0],), t).to(self.device)
                dt_base = torch.ones_like(t_vector).to(self.device) * math.log2(denoise_timesteps)


                with torch.no_grad():
                    v = self.forward(x, t_vector, dt_base, text_real)

                x = x + v * delta_t

                # log 8 steps
                if denoise_timesteps <= 8 or ti % (denoise_timesteps//8) ==0 or ti == denoise_timesteps-1:
                    with torch.no_grad():
                        decoded = self.vae.decode(x/self.vae.config.scaling_factor)[0]
                    
                    decoded = decoded.to("cpu")

                    all_x.append(decoded)

            if(len(all_x)==9):
                all_x = all_x[1:]
        
            # estimate FID metric
            decoded_denormalized = 255 * ((decoded - torch.min(decoded)) / (torch.max(decoded)-torch.min(decoded)+1e-8))
            
            # generated images
            self.fids[i].update(images.to(torch.uint8).to(self.device), real=True)
            self.fids[i].update(decoded_denormalized.to(torch.uint8).to(self.device), real=False)

            # log only a single batch of generated images and only on first device
            if self.trainer.is_global_zero and batch_idx == 0:

                all_x = torch.stack(all_x)

                def process_img(img):
                    # normalize in range [0,1]
                    img = img * 0.5 + 0.5
                    img = torch.clip(img, 0, 1)
                    img = img.permute(1,2,0)
                    return img
            
                fig, axs = plt.subplots(8, 8, figsize=(30,30))
                for t in range(min(8, all_x.shape[0])):
                    for j in range(8):        
                        axs[t, j].imshow(process_img(all_x[t, j]), vmin=0, vmax=1)
                
                
                fig.savefig(f"epoch_{self.trainer.current_epoch}_denoise_timesteps_{denoise_timesteps}.png")

                plt.close()

        return 0


    def on_validation_epoch_end(self):
        # if self.trainer.is_global_zero:
        for i in range(len(self.fids)):
            denoise_timesteps_i = self.denoise_timesteps[i]
            
            # Compute FID for the current timestep
            fid_val_i = self.fids[i].compute()
            self.fids[i].reset()


            # Log the FID value
            self.log(f"[FID] denoise_steps: {denoise_timesteps_i}", fid_val_i, on_epoch=True, on_step=False, sync_dist=True)


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

        return optimizer
        


    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only three channels by default
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# default model used in shortcut
def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}