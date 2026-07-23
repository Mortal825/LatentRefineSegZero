import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from training_scripts.refine_segzero.common import (
    answer_to_prompts,
    build_sam_image_tensor,
    compute_box_iou,
    compute_iou,
    dice_loss,
    extract_answer_json_and_think,
    find_subsequence,
    mask_to_box,
    scale_answer_to_image,
)
from training_scripts.refine_segzero.prompts import (
    DECISION_REFLECTION_TEMPLATE,
    GEOMETRIC_QUERY_TEMPLATE,
    REPAIR_REFLECTION_TEMPLATE,
)


DEFAULT_SAM2_CONFIG = "sam2_hiera_l.yaml"


@dataclass
class GenerationResult:
    output_text: str
    answer: Dict[str, Any]
    think: str
    generated_ids: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rms = hidden_states.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return hidden_states * rms * self.weight


# class PromptStabilizer(nn.Module):
#     def __init__(self, dim: int, norm_type: str = "layernorm", use_tanh: bool = True, init_scale: float = 0.05):
#         super().__init__()
#         self.norm = RMSNorm(dim) if norm_type.lower() == "rmsnorm" else nn.LayerNorm(dim)
#         self.use_tanh = use_tanh
#         safe_scale = max(float(init_scale), 1e-6)
#         gate_init = torch.log(torch.expm1(torch.tensor(safe_scale)))
#         self.scale_gate = nn.Parameter(gate_init.clone())

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.norm(hidden_states)
#         if self.use_tanh:
#             hidden_states = torch.tanh(hidden_states)
#         return hidden_states * F.softplus(self.scale_gate).to(hidden_states.dtype)


class DenseConnector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        prompt_dim: int,
        target_hw: Tuple[int, int],
        depth: int = 2,
        num_heads: int = 8,
        seed_channels: int = 256,
    ):
        super().__init__()
        self.prompt_dim = int(prompt_dim)
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))
        self.seed_channels = int(seed_channels)
        seed_h, seed_w = self.target_hw
        self.num_upsample_layers = 0
        while seed_h % 2 == 0 and seed_w % 2 == 0 and seed_h > 8 and seed_w > 8:
            seed_h //= 2
            seed_w //= 2
            self.num_upsample_layers += 1
        self.seed_hw = (seed_h, seed_w)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.seed_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.seed_channels * self.seed_hw[0] * self.seed_hw[1]),
        )
        upsample_layers: List[nn.Module] = []
        for _ in range(self.num_upsample_layers):
            upsample_layers.extend(
                [
                    nn.ConvTranspose2d(self.seed_channels, self.seed_channels, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(32, self.seed_channels),
                    nn.GELU(),
                ]
            )
        self.upsample = nn.Sequential(*upsample_layers)
        self.output_proj = nn.Conv2d(self.seed_channels, self.prompt_dim, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~hidden_mask.bool()
        encoded = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)
        encoded = encoded.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        denom = hidden_mask.sum(dim=1, keepdim=True).clamp_min(1).to(encoded.dtype)
        pooled = encoded.sum(dim=1) / denom
        dense = self.seed_proj(pooled).view(
            hidden_states.size(0),
            self.seed_channels,
            self.seed_hw[0],
            self.seed_hw[1],
        )
        dense = self.upsample(dense)
        dense = self.output_proj(dense)
        expected_hw = self.target_hw
        if dense.shape[-2:] != expected_hw:
            raise RuntimeError(f"DenseConnector output shape {dense.shape[-2:]} does not match target {expected_hw}")
        return dense

## 濠电偞鍨堕幐鍝ョ矓鐎电硶鍋撻崹顐€跨€规洏鍎甸、鏇㈠焺閸愩劍鏁梻浣告啞閻熴儵藝娴兼潙鏋侀柣鎰惈缁犳盯鐓崶銊﹀碍妞ゎ偅妫冮弻锝夊Ω閵夈儲鐤俹nnector Q_Fromer闁荤喐绮忛崺鍥垂閸︻厽顫?
class SparseConnector(nn.Module):
    def __init__(self, hidden_size: int, output_tokens: int, prompt_dim: int = 256, depth: int = 2, num_heads: int = 8):
        super().__init__()
        self.output_tokens = int(output_tokens)

        self.output_queries = nn.Parameter(torch.empty(1, self.output_tokens, hidden_size))
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, prompt_dim),
        )

    def forward(self, hidden_states: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~hidden_mask.bool()

        output_queries = self.output_queries.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).expand(hidden_states.size(0), -1, -1)

        attn_output, _ = self.cross_attn(
            query=output_queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        encoded = self.cross_attn_norm(output_queries + attn_output)
        encoded = self.encoder(encoded)

        return self.proj(encoded)
class TwoStageSparseConnector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_tokens: int,
        prompt_dim: int = 256,
        connector_dim: int = 1024,
        depth: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.prompt_dim = int(prompt_dim)
        self.connector_dim = int(connector_dim)

        self.output_queries = nn.Parameter(torch.empty(1, self.output_tokens, self.connector_dim))
        nn.init.normal_(self.output_queries, mean=0.0, std=0.02)

        self.hidden_cross_attn = nn.MultiheadAttention(
            embed_dim=self.connector_dim,
            num_heads=num_heads,
            kdim=hidden_size,
            vdim=hidden_size,
            dropout=0.0,
            batch_first=True,
        )
        self.hidden_cross_attn_norm = nn.LayerNorm(self.connector_dim)
        hidden_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.connector_dim,
            nhead=num_heads,
            dim_feedforward=self.connector_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.hidden_encoder = nn.TransformerEncoder(hidden_encoder_layer, num_layers=depth)

        self.prompt_cross_attn = nn.MultiheadAttention(
            embed_dim=self.prompt_dim,
            num_heads=num_heads,
            kdim=self.connector_dim,
            vdim=self.connector_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.prompt_cross_attn_norm = nn.LayerNorm(self.prompt_dim)
        prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.prompt_dim,
            nhead=num_heads,
            dim_feedforward=self.prompt_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(prompt_encoder_layer, num_layers=depth)

    def encode_hidden(self, hidden_states: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~hidden_mask.bool()
        output_queries = self.output_queries.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).expand(hidden_states.size(0), -1, -1)
        attn_output, _ = self.hidden_cross_attn(
            query=output_queries,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        encoded = self.hidden_cross_attn_norm(output_queries + attn_output)
        return self.hidden_encoder(encoded)

    def refine_prompt(self, prompt_queries: torch.Tensor, latent_tokens: torch.Tensor) -> torch.Tensor:
        prompt_queries = prompt_queries.to(device=latent_tokens.device, dtype=latent_tokens.dtype)
        if prompt_queries.size(1) == 0:
            prompt_queries = prompt_queries.new_zeros(prompt_queries.size(0), 1, self.prompt_dim)
        attn_output, _ = self.prompt_cross_attn(
            query=prompt_queries,
            key=latent_tokens,
            value=latent_tokens,
            need_weights=False,
        )
        encoded = self.prompt_cross_attn_norm(prompt_queries + attn_output)
        return self.prompt_encoder(encoded)

    def forward(self, hidden_states: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
        latent_tokens = self.encode_hidden(hidden_states, hidden_mask)
        prompt_queries = latent_tokens.new_zeros(latent_tokens.size(0), self.output_tokens, self.prompt_dim)
        return self.refine_prompt(prompt_queries, latent_tokens)
# class SparseConnector(nn.Module):
#     def __init__(self, hidden_size: int, output_tokens: int, prompt_dim: int = 256, depth: int = 2, num_heads: int = 8):
#         super().__init__()
#         self.output_tokens = int(output_tokens)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=num_heads,
#             dim_feedforward=hidden_size * 4,
#             dropout=0.0,
#             batch_first=True,
#             activation="gelu",
#             norm_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
#         self.proj = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, prompt_dim),
#         )

#     def forward(self, hidden_states: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
#         key_padding_mask = ~hidden_mask.bool()
#         encoded = self.encoder(hidden_states, src_key_padding_mask=key_padding_mask)
#         encoded = encoded.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
#         if encoded.size(1) < self.output_tokens:
#             pad = encoded.new_zeros(encoded.size(0), self.output_tokens - encoded.size(1), encoded.size(2))
#             encoded = torch.cat([encoded, pad], dim=1)
#         elif encoded.size(1) > self.output_tokens:
#             encoded = encoded[:, : self.output_tokens]
#         return self.proj(encoded)

## 闂備礁鎲￠悧鏇㈠箠鎼淬劌绠氶柛顐犲劚閻愬﹪鏌涢幘妤€鍠氶弳鐞眎dden states闂備焦鐪归崝宀勫吹婵笜nector闂備焦瀵х粙鎴︽儗閸屾稑顕遍柍鍝勬噹缁€鍌溾偓鐟板缁旀看M濠电偞鍨堕幐鍝ョ矓鐠虹尨鑰块柛娑欐綑閻?
class DirectCrossAttentionConnector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_tokens: int,
        prompt_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.output_tokens = int(output_tokens)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, prompt_dim),
        )

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        query_hidden_mask: torch.Tensor,
        image_hidden_states: torch.Tensor,
        image_hidden_mask: torch.Tensor,
    ) -> torch.Tensor:
        if image_hidden_states is None or image_hidden_mask is None:
            raise RuntimeError("DirectCrossAttentionConnector requires image hidden states.")

        if image_hidden_states.size(1) == 0:
            raise RuntimeError("Image hidden states are empty.")

        if not bool(image_hidden_mask.any().item()):
            raise RuntimeError("Image hidden mask has no valid tokens.")

        key_padding_mask = ~image_hidden_mask.bool()
        attn_output, _ = self.cross_attn(
            query=query_hidden_states,
            key=image_hidden_states,
            value=image_hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        fused_hidden = self.cross_attn_norm(query_hidden_states + attn_output)

        query_padding_mask = ~query_hidden_mask.bool()
        encoded = self.encoder(fused_hidden, src_key_padding_mask=query_padding_mask)
        encoded = encoded.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        if encoded.size(1) < self.output_tokens:
            pad = encoded.new_zeros(
                encoded.size(0),
                self.output_tokens - encoded.size(1),
                encoded.size(2),
            )
            encoded = torch.cat([encoded, pad], dim=1)
        elif encoded.size(1) > self.output_tokens:
            encoded = encoded[:, : self.output_tokens]

        return self.proj(encoded)

class RefineSegZeroModel(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        sam_model_path: str,
        processor_path: Optional[str] = None,
        sam_model_cfg: str = DEFAULT_SAM2_CONFIG,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
        aligned_output_tokens: int = 4,
        direct_output_tokens: int = 4,
        num_learnable_queries: int = 64,
        prompt_space: str = "sparse",
        hidden_state_layer: int = -2,
        connector_qformer_dim: int = 1024,
        freeze_mllm: bool = True,
        freeze_sam: bool = True,
    ):
        super().__init__()
        self.sam_model_cfg = sam_model_cfg
        self.hidden_state_layer = int(hidden_state_layer)
        self.prompt_space = str(prompt_space).lower()
        self.num_learnable_queries = int(num_learnable_queries)
        self.connector_qformer_dim = int(connector_qformer_dim)
        self.aligned_prompt_sampling_enabled = False
        self.aligned_prompt_jitter_enabled = False
        self.aligned_prompt_variants = ["box", "points", "box_points", "embedding"]
        self.aligned_prompt_variant_probs = [0.25, 0.25, 0.25, 0.25]
        self.aligned_box_jitter_ratio = 0.0
        self.aligned_point_jitter_ratio = 0.0
        self.aligned_soft_bbox_loss_weight = 0.0
        self.aligned_box_gap_loss_weight = 0.0
        self.aligned_prompt_embedding_loss_weight = 0.0
        self.has_direct_lora_adapter = False
        self.direct_lora_adapter_name = "direct"
        self.direct_lora_r = 0
        self.direct_lora_alpha = 0
        self.direct_lora_dropout = 0.0
        self.direct_lora_target_modules: List[str] = []
        self._direct_lora_active = False
        self._direct_lora_trainable = False
        model_path = processor_path or base_model_path
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        hidden_size = int(self.qwen_base_model().config.hidden_size)
        self.sam = self._build_sam_model(sam_model_path=sam_model_path, sam_model_cfg=sam_model_cfg, dtype=torch_dtype)
        prompt_dim = int(self.sam.sam_prompt_encoder.embed_dim)
        prompt_hw = tuple(int(v) for v in self.sam.sam_prompt_encoder.image_embedding_size)
        self.aligned_learnable_query = nn.Parameter(torch.randn(1, self.num_learnable_queries, hidden_size))
        self.direct_learnable_query = nn.Parameter(torch.empty(1, self.num_learnable_queries, hidden_size))
        nn.init.normal_(self.aligned_learnable_query, mean=0.0, std=0.02)
        with torch.no_grad():
            self.direct_learnable_query.copy_(self.aligned_learnable_query)
        if self.prompt_space == "dense":
            self.aligned_connector = DenseConnector(hidden_size=hidden_size, prompt_dim=prompt_dim, target_hw=prompt_hw).to(dtype=torch_dtype)
            self.direct_connector = DenseConnector(hidden_size=hidden_size, prompt_dim=prompt_dim, target_hw=prompt_hw).to(dtype=torch_dtype)
        elif self.prompt_space == "sparse":
            print("濠电偠鎻紞鈧繛澶嬫礋瀵偊藟缁涚禈rseConnector")
            self.aligned_connector = SparseConnector(hidden_size=hidden_size, output_tokens=aligned_output_tokens, prompt_dim=prompt_dim).to(dtype=torch_dtype)
            self.direct_connector = SparseConnector(hidden_size=hidden_size, output_tokens=direct_output_tokens, prompt_dim=prompt_dim).to(dtype=torch_dtype)
        elif self.prompt_space == "two_stage_sparse":
            print("Using TwoStageSparseConnector")
            self.aligned_connector = TwoStageSparseConnector(hidden_size=hidden_size, output_tokens=aligned_output_tokens, prompt_dim=prompt_dim, connector_dim=self.connector_qformer_dim).to(dtype=torch_dtype)
            self.direct_connector = TwoStageSparseConnector(hidden_size=hidden_size, output_tokens=direct_output_tokens, prompt_dim=prompt_dim, connector_dim=self.connector_qformer_dim).to(dtype=torch_dtype)
        elif self.prompt_space == "cross_sparse":
            print("濠电偠鎻紞鈧繛澶嬫礋瀵偊藟閻氼湼ectCrossAttentionConnector")
            self.aligned_connector = SparseConnector(hidden_size=hidden_size, output_tokens=aligned_output_tokens, prompt_dim=prompt_dim).to(dtype=torch_dtype)
            self.direct_connector = DirectCrossAttentionConnector(
                                                                hidden_size=hidden_size,
                                                                output_tokens=direct_output_tokens,
                                                                prompt_dim=prompt_dim,
                                                            ).to(dtype=torch_dtype)
        else:
            raise ValueError(f"Unsupported prompt_space: {self.prompt_space}")
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        if freeze_mllm:
            self.qwen.requires_grad_(False)
        if freeze_sam:
            self.sam.requires_grad_(False)
        self._cached_forward_hidden_states = None
        self._cached_forward_input_ids = None
        self._cached_forward_attention_mask = None

    def forward(self, batch: Dict[str, Any], train_branch: str = "both") -> Dict[str, Any]:
        return self.stage1_forward(batch=batch, train_branch=train_branch)

    def attach_direct_lora(
        self,
        r: int = 16,
        alpha: int = 64,
        dropout: float = 0.05,
        target_modules: Optional[Sequence[str]] = None,
        adapter_name: str = "direct",
    ) -> None:
        from peft import LoraConfig, get_peft_model

        modules = [str(item).strip() for item in (target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]) if str(item).strip()]
        if not modules:
            raise ValueError("direct LoRA requires at least one target module.")
        if self.has_direct_lora_adapter:
            raise RuntimeError("Direct LoRA adapter is already attached.")
        lora_config = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=modules,
            bias="none",
        )
        self.qwen = get_peft_model(self.qwen, lora_config, adapter_name=str(adapter_name))
        self.has_direct_lora_adapter = True
        self.direct_lora_adapter_name = str(adapter_name)
        self.direct_lora_r = int(r)
        self.direct_lora_alpha = int(alpha)
        self.direct_lora_dropout = float(dropout)
        self.direct_lora_target_modules = modules
        self.qwen.requires_grad_(False)
        self._direct_lora_trainable = True
        self._set_direct_lora_requires_grad(True)
        self._set_direct_lora_active(False)
        self._set_direct_lora_requires_grad(True)

    def load_direct_lora_adapter(
        self,
        adapter_path: str,
        adapter_name: str = "direct",
        is_trainable: bool = False,
    ) -> None:
        from peft import PeftModel

        path = Path(str(adapter_path))
        if not path.is_dir():
            raise FileNotFoundError(f"Direct LoRA adapter directory does not exist: {path}")
        resolved_path = path
        if not (resolved_path / "adapter_config.json").is_file():
            named_path = path / str(adapter_name)
            if (named_path / "adapter_config.json").is_file():
                resolved_path = named_path
            else:
                adapter_children = [child for child in path.iterdir() if child.is_dir() and (child / "adapter_config.json").is_file()]
                if len(adapter_children) == 1:
                    resolved_path = adapter_children[0]
        if not (resolved_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                "Direct LoRA adapter is missing adapter_config.json. "
                f"Checked {path} and child adapter directories."
            )
        if self.has_direct_lora_adapter:
            raise RuntimeError("Direct LoRA adapter is already attached.")
        self.qwen = PeftModel.from_pretrained(
            self.qwen,
            str(resolved_path),
            adapter_name=str(adapter_name),
            is_trainable=bool(is_trainable),
        )
        self.has_direct_lora_adapter = True
        self.direct_lora_adapter_name = str(adapter_name)
        self.qwen.requires_grad_(False)
        self._direct_lora_trainable = bool(is_trainable)
        self._set_direct_lora_requires_grad(bool(is_trainable))
        self._set_direct_lora_active(False)
        self._set_direct_lora_requires_grad(bool(is_trainable))

    def save_direct_lora_adapter(self, adapter_dir: Path) -> bool:
        if not self.has_direct_lora_adapter or not hasattr(self.qwen, "save_pretrained"):
            return False
        adapter_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {}
        if self.direct_lora_adapter_name:
            kwargs["selected_adapters"] = [self.direct_lora_adapter_name]
        self.qwen.save_pretrained(str(adapter_dir), **kwargs)
        return True

    def qwen_base_model(self) -> Any:
        if self.has_direct_lora_adapter and hasattr(self.qwen, "get_base_model"):
            return self.qwen.get_base_model()
        return self.qwen

    @property
    def qwen_device(self) -> torch.device:
        device = getattr(self.qwen, "device", None)
        if device is not None:
            return device
        return next(self.qwen.parameters()).device

    def get_qwen_base_model_for_export(self) -> Any:
        return self.qwen_base_model()

    def get_qwen_clean_base_state_dict_for_export(self) -> Dict[str, torch.Tensor]:
        clean_state: Dict[str, torch.Tensor] = {}
        for key, value in self.qwen_base_model().state_dict().items():
            if self._is_direct_lora_parameter_name(key):
                continue
            clean_key = str(key).replace(".base_layer.", ".")
            clean_state[clean_key] = value.detach().cpu()
        return clean_state

    def _is_direct_lora_parameter_name(self, name: str) -> bool:
        lowered = str(name).lower()
        return "lora_" in lowered or ".lora_a." in lowered or ".lora_b." in lowered

    def _set_direct_lora_requires_grad(self, enabled: bool) -> None:
        if not self.has_direct_lora_adapter:
            return
        for name, param in self.qwen.named_parameters():
            if self._is_direct_lora_parameter_name(name):
                param.requires_grad_(bool(enabled))

    def direct_lora_parameters(self) -> List[nn.Parameter]:
        if not self.has_direct_lora_adapter:
            return []
        return [param for name, param in self.qwen.named_parameters() if self._is_direct_lora_parameter_name(name)]

    def direct_lora_named_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        if not self.has_direct_lora_adapter:
            return []
        return [(name, param) for name, param in self.qwen.named_parameters() if self._is_direct_lora_parameter_name(name)]

    def freeze_aligned_branch(self) -> None:
        self.aligned_learnable_query.requires_grad_(False)
        self.aligned_connector.requires_grad_(False)

    def freeze_direct_branch(self) -> None:
        self.direct_learnable_query.requires_grad_(False)
        self.direct_connector.requires_grad_(False)
        self._set_direct_lora_requires_grad(False)

    def _set_direct_lora_active(self, active: bool) -> None:
        if not self.has_direct_lora_adapter:
            self._direct_lora_active = False
            return
        if active and hasattr(self.qwen, "set_adapter"):
            self.qwen.set_adapter(self.direct_lora_adapter_name)
        targets = [self.qwen]
        base_model = getattr(self.qwen, "base_model", None)
        if base_model is not None:
            targets.append(base_model)
        method_name = "enable_adapter_layers" if active else "disable_adapter_layers"
        for target in targets:
            method = getattr(target, method_name, None)
            if callable(method):
                method()
        self._direct_lora_active = bool(active)
        if self._direct_lora_trainable:
            self._set_direct_lora_requires_grad(True)

    @contextmanager
    def direct_lora_enabled(self):
        previous = bool(self._direct_lora_active)
        self._set_direct_lora_active(True)
        try:
            yield
        finally:
            self._set_direct_lora_active(previous)

    @contextmanager
    def direct_lora_disabled(self):
        previous = bool(self._direct_lora_active)
        self._set_direct_lora_active(False)
        try:
            yield
        finally:
            self._set_direct_lora_active(previous)


    def _build_sam_model(self, sam_model_path: str, sam_model_cfg: str, dtype: torch.dtype):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError("sam2 is required for refine SegZero.") from exc
        sam2_model = build_sam2(sam_model_cfg, sam_model_path, device="cuda", dtype=dtype)
        predictor = SAM2ImagePredictor(sam2_model)
        self._sam_predictor = predictor
        self._sam_transforms = predictor._transforms
        self._sam_mask_threshold = predictor.mask_threshold
        return predictor.model

    def geometric_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if name in {"aligned_learnable_query", "direct_learnable_query"}
            or name.startswith("aligned_connector.")
            or name.startswith("direct_connector.")
        }
        state["learnable_query"] = self.aligned_learnable_query.detach().cpu()
        return state

    def stage2_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.geometric_state_dict()

    def load_custom_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        migrated = dict(state_dict)
        legacy_query = migrated.get("learnable_query")
        if legacy_query is not None:
            migrated.setdefault("aligned_learnable_query", legacy_query)
            migrated.setdefault("direct_learnable_query", legacy_query)
        if not strict:
            current_state = self.state_dict()
            skipped = []
            for key in list(migrated.keys()):
                value = migrated[key]
                if key in current_state and tuple(value.shape) != tuple(current_state[key].shape):
                    skipped.append(key)
                    migrated.pop(key)
            if skipped:
                print(
                    "[RefineSegZeroModel] skipped incompatible state keys: "
                    + ", ".join(skipped[:8])
                    + (f" ... (+{len(skipped) - 8} more)" if len(skipped) > 8 else ""),
                    flush=True,
                )
        return self.load_state_dict(migrated, strict=strict)

    @property
    def learnable_query(self) -> nn.Parameter:
        return self.aligned_learnable_query

    def _select_learnable_query(self, branch: Optional[str] = None) -> nn.Parameter:
        branch_key = str(branch or "aligned").strip().lower()
        if branch_key in {"", "both", "aligned", "compat"}:
            return self.aligned_learnable_query
        if branch_key == "direct":
            return self.direct_learnable_query
        raise ValueError(f"Unsupported learnable query branch: {branch}")

    def build_generation_messages(self, image: Image.Image, question: str, resize_size: int) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
                    {"type": "text", "text": GEOMETRIC_QUERY_TEMPLATE.format(Question=question.lower().strip("."))},
                ],
            }
        ]

    def build_decision_messages(
        self,
        image: Image.Image,
        question: str,
        first_answer_text: str,
        mask_summary: str,
        resize_size: int,
        reflection_mask_context_mode: str = "image_crop",
        mask_image: Optional[Image.Image] = None,
        crop_image: Optional[Image.Image] = None,
        mask_resize_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        mode = str(reflection_mask_context_mode).lower()
        content: List[Dict[str, Any]] = [
            {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
        ]
        if mode in {"image_mask", "mask_image", "mask_image_crop", "text_mask_crop"} and mask_image is not None:
            target_size = int(mask_resize_size or resize_size)
            content.append({"type": "image", "image": mask_image.resize((target_size, target_size), Image.NEAREST)})
        if mode == "image_crop" and crop_image is not None:
            target_size = int(mask_resize_size or resize_size)
            content.append({"type": "image", "image": crop_image.resize((target_size, target_size), Image.BILINEAR)})
        if mode in {"mask_image_crop", "text_mask_crop"} and crop_image is not None:
            content.append({"type": "image", "image": crop_image.resize((resize_size, resize_size), Image.BILINEAR)})
        context_note = "The first image is the original image."
        if mode == "image_crop":
            context_note += " The second image is the segmented object crop from the first-pass result."
        elif mode == "image_mask":
            context_note += " The second image is the first-pass predicted binary mask."
        elif mode == "mask_image":
            context_note += " The second image is the first-pass mask overlay."
        elif mode in {"mask_image_crop", "text_mask_crop"}:
            context_note += " The second image is the first-pass mask overlay. The third image is the cropped region from that mask."
        if mode in {"text", "text_mask_crop"}:
            context_note += f" Mask summary: {mask_summary}"
        return [
            {
                "role": "user",
                "content": content
                + [
                    {
                        "type": "text",
                        "text": context_note
                        + "\n"
                        + DECISION_REFLECTION_TEMPLATE.format(
                            question=question,
                            first_answer=first_answer_text,
                            mask_summary=mask_summary if mode in {"text", "text_mask_crop"} else "provided as visual context",
                        ),
                    },
                ],
            }
        ]

    def build_repair_messages(
        self,
        image: Image.Image,
        question: str,
        first_answer_text: str,
        decision_answer: str,
        mask_summary: str,
        resize_size: int,
        reflection_mask_context_mode: str = "image_crop",
        mask_image: Optional[Image.Image] = None,
        crop_image: Optional[Image.Image] = None,
        mask_resize_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        mode = str(reflection_mask_context_mode).lower()
        content: List[Dict[str, Any]] = [
            {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
        ]
        if mode in {"image_mask", "mask_image", "mask_image_crop", "text_mask_crop"} and mask_image is not None:
            target_size = int(mask_resize_size or resize_size)
            content.append({"type": "image", "image": mask_image.resize((target_size, target_size), Image.NEAREST)})
        if mode == "image_crop" and crop_image is not None:
            target_size = int(mask_resize_size or resize_size)
            content.append({"type": "image", "image": crop_image.resize((target_size, target_size), Image.BILINEAR)})
        if mode in {"mask_image_crop", "text_mask_crop"} and crop_image is not None:
            content.append({"type": "image", "image": crop_image.resize((resize_size, resize_size), Image.BILINEAR)})
        context_note = "The first image is the original image."
        if mode == "image_crop":
            context_note += " The second image is the segmented object crop from the first-pass result."
        elif mode == "image_mask":
            context_note += " The second image is the first-pass predicted binary mask."
        elif mode == "mask_image":
            context_note += " The second image is the first-pass mask overlay."
        elif mode in {"mask_image_crop", "text_mask_crop"}:
            context_note += " The second image is the first-pass mask overlay. The third image is the cropped region from that mask."
        if mode in {"text", "text_mask_crop"}:
            context_note += f" Mask summary: {mask_summary}"
        return [
            {
                "role": "user",
                "content": content
                + [
                    {
                        "type": "text",
                        "text": context_note
                        + "\n"
                        + REPAIR_REFLECTION_TEMPLATE.format(
                            question=question,
                            first_answer=first_answer_text,
                            decision_answer=decision_answer,
                            mask_summary=mask_summary if mode in {"text", "text_mask_crop"} else "provided as visual context",
                        ),
                    },
                ],
            }
        ]

    def _processor_inputs(self, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = process_vision_info([messages])
        return self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    def _single_sample_inputs(self, model_inputs: Dict[str, torch.Tensor], index: int) -> Dict[str, torch.Tensor]:
        sample_inputs: Dict[str, torch.Tensor] = {}
        batch_size = model_inputs["input_ids"].size(0)
        for key, value in model_inputs.items():
            if not torch.is_tensor(value):
                continue
            if value.dim() > 0 and value.size(0) == batch_size:
                sample_inputs[key] = value[index : index + 1]
            else:
                sample_inputs[key] = value
        return sample_inputs

    def _move_inputs_to_device(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.qwen_device) if torch.is_tensor(value) else value for key, value in model_inputs.items()}

    def _prepare_inputs_with_learnable_queries(
        self,
        model_inputs: Dict[str, torch.Tensor],
        branch: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        device_inputs = self._move_inputs_to_device(model_inputs)
        input_ids = device_inputs["input_ids"]
        attention_mask = device_inputs["attention_mask"]
        inputs_embeds = self.qwen_base_model().model.embed_tokens(input_ids)
        pixel_values = device_inputs.get("pixel_values")
        image_grid_thw = device_inputs.get("image_grid_thw")
        if pixel_values is not None:
            visual_dtype = getattr(self.qwen_base_model().visual, "dtype", None)
            if visual_dtype is None:
                visual_dtype = next(self.qwen_base_model().visual.parameters()).dtype
            pixel_values = pixel_values.to(dtype=visual_dtype)
            image_embeds = self.qwen_base_model().visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (
                (input_ids == self.qwen_base_model().config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        query_param = self._select_learnable_query(branch)
        learnable_query = query_param.to(inputs_embeds.device, inputs_embeds.dtype).repeat(inputs_embeds.size(0), 1, 1)
        inputs_embeds = torch.cat([inputs_embeds, learnable_query], dim=1)
        query_mask = torch.ones(attention_mask.size(0), self.num_learnable_queries, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, query_mask], dim=1)
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

    def _generate_once(
        self,
        model_inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> Tuple[List[GenerationResult], Any]:
        device_inputs = self._move_inputs_to_device(model_inputs)
        generate_kwargs = dict(
            **device_inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        with torch.set_grad_enabled(self.qwen.training):
            outputs = self.qwen.generate(**generate_kwargs)
        generated = outputs.sequences
        input_len = int(device_inputs["input_ids"].shape[1])
        trimmed = generated[:, input_len:]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results: List[GenerationResult] = []
        for seq, text in zip(trimmed, texts):
            try:
                answer, think, _ = extract_answer_json_and_think(text)
            except Exception:
                answer, think = {"bbox": [0, 0, 0, 0], "points_1": [0, 0], "points_2": [0, 0]}, ""
            results.append(GenerationResult(output_text=text, answer=answer, think=think, generated_ids=seq.detach().cpu()))
        return results, outputs

    # def _forward_hidden_states(self, model_inputs: Dict[str, torch.Tensor], require_grad: bool) -> torch.Tensor:
    #     device_inputs = self._prepare_inputs_with_learnable_queries(model_inputs)
    #     context = torch.enable_grad() if require_grad else torch.no_grad()
    #     with context:
    #         outputs = self.qwen(
    #             input_ids=None,
    #             inputs_embeds=device_inputs["inputs_embeds"],
    #             attention_mask=device_inputs["attention_mask"],
    #             output_hidden_states=True,
    #             use_cache=False,
    #             return_dict=True,
    #         )
    #     return outputs.hidden_states[self.hidden_state_layer]
    ## 闂備胶顢婂▔娑㈡晝閿曗偓鐓ら柛褎顨呯粻鎶芥煏婢跺牆濡跨紒鐘崇叀閺?hidden states 闂備焦鐪归崝宀€鈧凹鍓熷畷顖炲箻閺傘儲鐏冮梺鍝勬川閸犳劙寮抽銏＄厵閻庢稒锚婵洭鏌涘Ο鑽ゃ€掔紒杈ㄦ崌楠炴ê鐣烽崶銊﹀枛闂備礁鎲￠崝鏇犵矓妞嬪孩宕查柟杈剧畱閻愬﹪鏌ｉ幋婵囧窛缂佲偓婢舵劖鐓涢柛宀€鍋愰崑鎾斥槈濮橈絾鏅欑紓鍌氬€烽悞锕傗€︾槐鍍秙sConnector
    

    ## 闂備礁鎲￠悷褏浜搁悤绱刱闂備礁鎲￠懝鎯归悜鑺ュ仺闁煎鍊撳▽顏堟煕瀹€瀣洭缂佲偓婢舵劖鐓犻柡澶婄仢椤ㄦ瑧绱掑Δ鈧ˇ闈涱嚕娴兼惌鏁嶆繛鍡樺焾娴煎洭鎮楀▓鍨灀闁稿鎸搁埞鎴︻敊绾板崬鍓板┑顔角氶弳鏄k
    def _make_late_attention_drop_pre_hook(self,drop_positions: torch.Tensor):
        drop_positions = drop_positions.detach().long().cpu()

        def hook(module, args, output):
            if drop_positions.numel() == 0:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                if hidden_states is None:
                    return output
                pos = drop_positions.to(hidden_states.device)
                hidden_states = hidden_states.clone()
                hidden_states[:, pos, :] = 0
                return (hidden_states,) + output[1:]

            hidden_states = output
            if hidden_states is None:
                return output
            pos = drop_positions.to(hidden_states.device)
            hidden_states = hidden_states.clone()
            hidden_states[:, pos, :] = 0
            return hidden_states

        return hook

    
    def _forward_hidden_states(
        self,
        model_inputs: Dict[str, torch.Tensor],
        require_grad: bool,
        branch: Optional[str] = None,
    ) -> torch.Tensor:
        device_inputs = self._prepare_inputs_with_learnable_queries(model_inputs, branch=branch)
        raw_device_inputs = self._move_inputs_to_device(model_inputs)
        context = torch.enable_grad() if require_grad else torch.no_grad()

        late_cfg = getattr(self, "_late_attention_drop", None)
        hook_handles = []
        if late_cfg:
            start_layer = int(late_cfg.get("start_layer", -10))
            drop_positions = late_cfg.get("drop_positions")
            layers = getattr(self.qwen_base_model().model, "layers", None)
            if layers is not None and drop_positions is not None:
                num_layers = len(layers)
                if start_layer < 0:
                    start_layer = max(num_layers + start_layer, 0)
                start_layer = min(max(start_layer, 0), num_layers)
                hook_fn = self._make_late_attention_drop_pre_hook(drop_positions)
                for layer in layers[start_layer:]:
                    print("register late attention drop hook")
                    hook_handles.append(layer.register_forward_hook(hook_fn))


        branch_key = str(branch or "aligned").strip().lower()
        if self.training and self.has_direct_lora_adapter:
            lora_context = self.direct_lora_enabled() if branch_key == "direct" else self.direct_lora_disabled()
        else:
            lora_context = nullcontext()

        with lora_context:
            with context:
                outputs = self.qwen(
                    input_ids=None,
                    inputs_embeds=device_inputs["inputs_embeds"],
                    attention_mask=device_inputs["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
        for handle in hook_handles:
            handle.remove()
        hidden_states = outputs.hidden_states[self.hidden_state_layer]
        self._cached_forward_hidden_states = hidden_states
        self._cached_forward_input_ids = raw_device_inputs["input_ids"]
        self._cached_forward_attention_mask = raw_device_inputs["attention_mask"]
        return hidden_states


    def _find_question_span(self, valid_ids: torch.Tensor, question: str) -> Tuple[int, int]:
        target_ids = self.processor.tokenizer(question.lower().strip("."), add_special_tokens=False)["input_ids"]
        start_idx, end_idx = find_subsequence(valid_ids.tolist(), target_ids)
        if start_idx >= 0:
            return start_idx, end_idx
        non_image_positions = [
            idx
            for idx, token_id in enumerate(valid_ids.tolist())
            if token_id != getattr(self.qwen_base_model().config, "image_token_id", -1)
        ]
        if not non_image_positions:
            return 0, int(valid_ids.numel())
        end_idx = non_image_positions[-1] + 1
        start_idx = max(end_idx - max(len(target_ids), 8), 0)
        return start_idx, end_idx

    def extract_question_hidden_states(
        self,
        model_inputs: Dict[str, torch.Tensor],
        questions: Sequence[str],
        require_grad: bool = False,
        branch: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self._forward_hidden_states(model_inputs, require_grad=require_grad, branch=branch)
        device_inputs = self._move_inputs_to_device(model_inputs)
        input_ids = device_inputs["input_ids"]
        attention_mask = device_inputs["attention_mask"]
        per_sample: List[torch.Tensor] = []
        lengths: List[int] = []
        for idx, question in enumerate(questions):
            valid_len = int(attention_mask[idx].sum().item())
            valid_ids = input_ids[idx, :valid_len]
            start_idx, end_idx = self._find_question_span(valid_ids, question)
            seq_hidden = hidden_states[idx, start_idx:end_idx, :]
            if seq_hidden.size(0) == 0:
                seq_hidden = hidden_states[idx, valid_len - 1 : valid_len, :]
            per_sample.append(seq_hidden)
            lengths.append(int(seq_hidden.size(0)))
        max_len = max(lengths) if lengths else 1
        batch_size = len(per_sample)
        hidden_size = hidden_states.size(-1)
        padded = hidden_states.new_zeros((batch_size, max_len, hidden_size))
        mask = torch.zeros((batch_size, max_len), device=hidden_states.device, dtype=torch.bool)
        for idx, seq_hidden in enumerate(per_sample):
            seq_len = seq_hidden.size(0)
            padded[idx, :seq_len] = seq_hidden
            mask[idx, :seq_len] = True
        return padded, mask

    def extract_query_hidden_states(
        self,
        model_inputs: Dict[str, torch.Tensor],
        require_grad: bool = False,
        branch: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self._forward_hidden_states(model_inputs, require_grad=require_grad, branch=branch)
        query_hidden = hidden_states[:, -self.num_learnable_queries :, :]
        query_mask = torch.ones(
            query_hidden.size(0),
            query_hidden.size(1),
            device=query_hidden.device,
            dtype=torch.bool,
        )
        return query_hidden, query_mask
    ## 闂備礁鎼崐鐟邦熆濮椻偓璺柛鎰靛枛缁犵敻鏌熼柇锕€澧紒?image hidden 闂備焦鐪归崝宀€鈧凹鍓熷畷鐢告焼瀹ュ懏鐎?
    def _extract_image_hidden_states_from_cached_forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self._cached_forward_hidden_states
        input_ids = self._cached_forward_input_ids
        attention_mask = self._cached_forward_attention_mask

        if hidden_states is None or input_ids is None or attention_mask is None:
            raise RuntimeError("Forward hidden-state cache is empty. Run _forward_hidden_states first.")

        image_token_id = getattr(self.qwen_base_model().config, "image_token_id", -1)
        per_sample: List[torch.Tensor] = []
        lengths: List[int] = []

        for idx in range(input_ids.size(0)):
            valid_len = int(attention_mask[idx].sum().item())
            valid_ids = input_ids[idx, :valid_len]
            valid_hidden = hidden_states[idx, :valid_len, :]
            image_mask = valid_ids == image_token_id
            seq_hidden = valid_hidden[image_mask]
            per_sample.append(seq_hidden)
            lengths.append(int(seq_hidden.size(0)))

        max_len = max(lengths) if lengths else 1
        batch_size = len(per_sample)
        hidden_size = hidden_states.size(-1)
        padded = hidden_states.new_zeros((batch_size, max_len, hidden_size))
        mask = torch.zeros((batch_size, max_len), device=hidden_states.device, dtype=torch.bool)

        for idx, seq_hidden in enumerate(per_sample):
            seq_len = seq_hidden.size(0)
            if seq_len > 0:
                padded[idx, :seq_len] = seq_hidden
                mask[idx, :seq_len] = True

        return padded, mask


    def build_connector_embeddings(
        self,
        query_hidden_batch: torch.Tensor,
        query_hidden_mask: torch.Tensor,
        branch: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_key = str(branch or "both").strip().lower()
        if branch_key not in {"both", "aligned", "direct"}:
            raise ValueError(f"Unsupported connector branch: {branch}")

        aligned_embedding = None
        direct_embedding = None
        if self.prompt_space == "sparse":
            if branch_key in {"both", "aligned"}:
                aligned_embedding = self.aligned_connector(query_hidden_batch, query_hidden_mask)
            if branch_key in {"both", "direct"}:
                direct_embedding = self.direct_connector(query_hidden_batch, query_hidden_mask)
        elif self.prompt_space == "two_stage_sparse":
            if branch_key in {"both", "aligned"}:
                aligned_embedding = self.aligned_connector.encode_hidden(query_hidden_batch, query_hidden_mask)
            if branch_key in {"both", "direct"}:
                direct_embedding = self.direct_connector.encode_hidden(query_hidden_batch, query_hidden_mask)
        elif self.prompt_space == "cross_sparse":
            image_hidden_batch, image_hidden_mask = self._extract_image_hidden_states_from_cached_forward()
            if branch_key in {"both", "aligned"}:
                aligned_embedding = self.aligned_connector(query_hidden_batch, query_hidden_mask)
            if branch_key in {"both", "direct"}:
                direct_embedding = self.direct_connector(
                    query_hidden_batch,
                    query_hidden_mask,
                    image_hidden_batch,
                    image_hidden_mask,
                )
        elif self.prompt_space == "dense":
            if branch_key in {"both", "aligned"}:
                aligned_embedding = self.aligned_connector(query_hidden_batch, query_hidden_mask)
            if branch_key in {"both", "direct"}:
                direct_embedding = self.direct_connector(query_hidden_batch, query_hidden_mask)
        else:
            raise ValueError(f"Unsupported prompt_space: {self.prompt_space}")
        return aligned_embedding, direct_embedding

    def _prepare_prompt_inputs(
        self,
        sam_images: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
        input_boxes: Optional[List[List[int]]] = None,
        input_points: Optional[List[List[List[int]]]] = None,
        refiner_points: Optional[List[List[List[int]]]] = None,
        extra_sparse: Optional[torch.Tensor] = None,
        dense_prompt_embeddings: Optional[torch.Tensor] = None,
        prompt_refiner: Optional[TwoStageSparseConnector] = None,
        use_points: bool = True,
        mode = "aligned",
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any], List[Dict[str, Any]]]:
        with torch.no_grad():
            backbone_out = self.sam.forward_image(sam_images)
            _, image_embeddings, _, _ = self.sam._prepare_backbone_features(backbone_out)
            image_embeddings = [feature.to(sam_images.dtype) for feature in image_embeddings]
            if self.sam.directly_add_no_mem_embed:
                image_embeddings[-1] = image_embeddings[-1] + self.sam.no_mem_embed
        batch_size = sam_images.shape[0]
        feats = [
            feature.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feature, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        sparse_list: List[torch.Tensor] = []
        native_sparse_list: List[torch.Tensor] = []
        dense_list: List[torch.Tensor] = []
        prompt_aux_list: List[Dict[str, Any]] = []
        for idx in range(batch_size):
            concat_points = None
            point_only_prompt = None
            refiner_point_prompt = None
            connector_sparse = None
            if input_points is not None and idx < len(input_points) and input_points[idx]:
                point_tensor = torch.tensor(input_points[idx], device=sam_images.device, dtype=torch.float32)
                point_tensor = self._sam_transforms.transform_coords(point_tensor, normalize=True, orig_hw=image_sizes[idx])
                point_labels = torch.ones((len(input_points[idx]),), device=sam_images.device, dtype=torch.int32)
                point_only_prompt = (point_tensor.unsqueeze(0), point_labels.unsqueeze(0))
                if use_points:
                    concat_points = point_only_prompt
            if refiner_points is not None and idx < len(refiner_points) and refiner_points[idx]:
                refiner_point_tensor = torch.tensor(refiner_points[idx], device=sam_images.device, dtype=torch.float32)
                refiner_point_tensor = self._sam_transforms.transform_coords(refiner_point_tensor, normalize=True, orig_hw=image_sizes[idx])
                refiner_point_labels = torch.ones((len(refiner_points[idx]),), device=sam_images.device, dtype=torch.int32)
                refiner_point_prompt = (refiner_point_tensor.unsqueeze(0), refiner_point_labels.unsqueeze(0))
            if input_boxes is not None and idx < len(input_boxes) and input_boxes[idx]:
                prompt_boxes = torch.tensor(input_boxes[idx], device=sam_images.device, dtype=torch.float32).unsqueeze(0)
                prompt_boxes = self._sam_transforms.transform_boxes(prompt_boxes, normalize=True, orig_hw=image_sizes[idx])
                box_coords = prompt_boxes.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=sam_images.device)
                if concat_points is not None:
                    concat_points = (
                        torch.cat([box_coords, concat_points[0]], dim=1),
                        torch.cat([box_labels, concat_points[1]], dim=1),
                    )
                else:
                    concat_points = (box_coords, box_labels)
            sparse_embeddings, native_dense_embeddings = self.sam.sam_prompt_encoder(points=concat_points, boxes=None, masks=None)
            native_sparse_embeddings = sparse_embeddings
            refiner_query_embeddings = native_sparse_embeddings
            if concat_points is not None:
                refiner_query_embeddings = refiner_query_embeddings[:, : concat_points[0].size(1)]
            if refiner_point_prompt is not None:
                refiner_query_embeddings, _ = self.sam.sam_prompt_encoder(points=refiner_point_prompt, boxes=None, masks=None)
                refiner_query_embeddings = refiner_query_embeddings[:, : len(refiner_points[idx])]
            elif point_only_prompt is not None:
                refiner_query_embeddings, _ = self.sam.sam_prompt_encoder(points=point_only_prompt, boxes=None, masks=None)
                refiner_query_embeddings = refiner_query_embeddings[:, : len(input_points[idx])]
            # print("xxixixi")
            # print("prompt_refiner",prompt_refiner,"extra_sparse",extra_sparse)
            if prompt_refiner is not None and extra_sparse is not None:
                # print("refiner_query_embeddings",refiner_query_embeddings.shape)
                connector_sparse = prompt_refiner.refine_prompt(refiner_query_embeddings, extra_sparse[idx : idx + 1])
                if mode == "aligned":
                    # print("aligned_native_sparse_embeddings",native_sparse_embeddings.shape)
                    # print("aligned_connector_sparse",connector_sparse.shape)
                    sparse_embeddings = torch.cat([native_sparse_embeddings.to(connector_sparse.dtype), connector_sparse], dim=1)
                    # print("aligned_sparse_embeddings",sparse_embeddings.shape)
                    # sparse_embeddings = native_sparse_embeddings.to(connector_sparse.dtype)
                elif mode == "direct":
                    sparse_embeddings = connector_sparse
                else:
                    sparse_embeddings = connector_sparse
            elif extra_sparse is not None and mode == "aligned":
                sparse_embeddings = torch.cat([sparse_embeddings, extra_sparse[idx : idx + 1]], dim=1)
            elif extra_sparse is not None and mode == "direct":
                sparse_embeddings = sparse_embeddings + extra_sparse[idx : idx + 1]
            sparse_list.append(sparse_embeddings)
            native_sparse_list.append(native_sparse_embeddings)
            dense_list.append(native_dense_embeddings)
            prompt_aux_list.append(
                {
                    "connector_sparse": connector_sparse,
                    "refiner_query_embeddings": refiner_query_embeddings,
                }
            )
        if dense_prompt_embeddings is not None:
            dense_list = [dense_prompt_embeddings[idx : idx + 1] for idx in range(batch_size)]
        return sparse_list, native_sparse_list, dense_list, features, prompt_aux_list

    def _decode_masks(
        self,
        features: Dict[str, Any],
        sparse_list: List[torch.Tensor],
        dense_list: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        pred_masks: List[torch.Tensor] = []
        for idx, sparse_embeddings in enumerate(sparse_list):
            high_res_features = [feat_level[idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
            low_res_masks, _, _, _ = self.sam.sam_mask_decoder(
                image_embeddings=features["image_embed"][idx].unsqueeze(0),
                image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_list[idx],
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            pred_mask = self._sam_transforms.postprocess_masks(low_res_masks.float(), tuple(image_sizes[idx]))
            pred_masks.append(pred_mask)
        return pred_masks

    def _gt_box_tensor(
        self,
        gt_box: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor([float(v) for v in gt_box[:4]], device=device, dtype=dtype)

    def _soft_box_from_logits(self, pred_mask_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(pred_mask_logits.float())
        height, width = int(probs.shape[-2]), int(probs.shape[-1])
        eps = 1.0e-6
        x_coords = torch.arange(width, device=probs.device, dtype=probs.dtype)
        y_coords = torch.arange(height, device=probs.device, dtype=probs.dtype)
        col_mass = probs.sum(dim=0).clamp_min(eps)
        row_mass = probs.sum(dim=1).clamp_min(eps)
        x_temp = max(float(width) * 0.02, 1.0)
        y_temp = max(float(height) * 0.02, 1.0)
        x_log_mass = torch.log(col_mass)
        y_log_mass = torch.log(row_mass)
        x1 = (torch.softmax(x_log_mass - x_coords / x_temp, dim=0) * x_coords).sum()
        x2 = (torch.softmax(x_log_mass + x_coords / x_temp, dim=0) * x_coords).sum()
        y1 = (torch.softmax(y_log_mass - y_coords / y_temp, dim=0) * y_coords).sum()
        y2 = (torch.softmax(y_log_mass + y_coords / y_temp, dim=0) * y_coords).sum()
        return torch.stack([x1, y1, x2, y2])

    def _soft_box_iou(self, pred_box: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
        x1 = torch.minimum(pred_box[0], pred_box[2])
        y1 = torch.minimum(pred_box[1], pred_box[3])
        x2 = torch.maximum(pred_box[0], pred_box[2])
        y2 = torch.maximum(pred_box[1], pred_box[3])
        gx1 = torch.minimum(gt_box[0], gt_box[2])
        gy1 = torch.minimum(gt_box[1], gt_box[3])
        gx2 = torch.maximum(gt_box[0], gt_box[2])
        gy2 = torch.maximum(gt_box[1], gt_box[3])
        inter_w = (torch.minimum(x2, gx2) - torch.maximum(x1, gx1)).clamp_min(0.0)
        inter_h = (torch.minimum(y2, gy2) - torch.maximum(y1, gy1)).clamp_min(0.0)
        inter = inter_w * inter_h
        pred_area = (x2 - x1).clamp_min(0.0) * (y2 - y1).clamp_min(0.0)
        gt_area = (gx2 - gx1).clamp_min(0.0) * (gy2 - gy1).clamp_min(0.0)
        return inter / (pred_area + gt_area - inter).clamp_min(1.0e-6)

    def _normalized_soft_box_l1(
        self,
        pred_box: torch.Tensor,
        gt_box: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        norm = pred_box.new_tensor([max(width, 1), max(height, 1), max(width, 1), max(height, 1)])
        return (pred_box - gt_box).abs().div(norm).mean()

    def _compute_box_gap_loss(
        self,
        stage1_box: Optional[Sequence[int]],
        gt_box: Sequence[int],
        soft_bbox_iou: torch.Tensor,
    ) -> torch.Tensor:
        if stage1_box is None:
            return soft_bbox_iou.new_zeros(())
        try:
            if len(stage1_box) < 4:
                return soft_bbox_iou.new_zeros(())
            baseline_iou = compute_box_iou(stage1_box, gt_box)
        except Exception:
            return soft_bbox_iou.new_zeros(())
        return torch.relu(soft_bbox_iou.new_tensor(float(baseline_iou)) - soft_bbox_iou)

    def _compute_prompt_embedding_loss(
        self,
        connector_sparse: Optional[torch.Tensor],
        gt_point: Optional[Sequence[int]],
        refiner_points: Optional[Sequence[Sequence[int]]],
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if connector_sparse is None:
            zero = self.aligned_learnable_query.sum() * 0.0
            return zero, zero
        zero = connector_sparse.sum() * 0.0
        if connector_sparse.ndim != 3 or connector_sparse.size(1) <= 0:
            return zero, zero
        if not gt_point or len(gt_point) < 2:
            return zero, zero
        try:
            gt_x, gt_y = int(round(float(gt_point[0]))), int(round(float(gt_point[1])))
        except Exception:
            return zero, zero

        height, width = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            return zero, zero
        gt_x = max(0, min(gt_x, width - 1))
        gt_y = max(0, min(gt_y, height - 1))

        token_idx = 0
        prompt_points = self._clip_prompt_points(refiner_points, image_size)
        if prompt_points:
            distances = [
                (float(point[0]) - float(gt_x)) ** 2 + (float(point[1]) - float(gt_y)) ** 2
                for point in prompt_points
            ]
            token_idx = int(min(range(len(distances)), key=lambda idx: distances[idx]))
        token_idx = max(0, min(token_idx, connector_sparse.size(1) - 1))

        with torch.no_grad():
            point_tensor = torch.tensor([[gt_x, gt_y]], device=connector_sparse.device, dtype=torch.float32)
            point_tensor = self._sam_transforms.transform_coords(point_tensor, normalize=True, orig_hw=image_size)
            point_labels = torch.ones((1,), device=connector_sparse.device, dtype=torch.int32)
            target_sparse, _ = self.sam.sam_prompt_encoder(
                points=(point_tensor.unsqueeze(0), point_labels.unsqueeze(0)),
                boxes=None,
                masks=None,
            )
            if target_sparse.ndim != 3 or target_sparse.size(1) <= 0:
                return zero, zero
            target_token = target_sparse[0, 0].detach()

        pred_token = connector_sparse[0, token_idx]
        if pred_token.numel() != target_token.numel():
            return zero, zero
        loss = F.smooth_l1_loss(pred_token.float(), target_token.to(pred_token.device).float())
        return loss.to(connector_sparse.dtype), connector_sparse.new_tensor(1.0)

    def _compute_mask_bbox_losses(
        self,
        pred_mask_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_box: Sequence[int],
        gt_box: Sequence[int],
        width: int,
        height: int,
    ) -> Dict[str, torch.Tensor]:
        gt_tensor = gt_mask.unsqueeze(0).unsqueeze(0).to(pred_mask_logits.device, pred_mask_logits.dtype)
        pred_tensor = pred_mask_logits.unsqueeze(0).unsqueeze(0)
        bce = F.binary_cross_entropy_with_logits(pred_tensor, gt_tensor)
        dice = dice_loss(pred_tensor, gt_tensor).mean()
        bbox_iou_error = 1.0 - compute_box_iou(pred_box, gt_box)
        bbox_l1 = sum(
            abs(float(a) - float(b)) / float(max(n, 1))
            for a, b, n in zip(pred_box, gt_box, [width, height, width, height])
        ) / 4.0
        bbox_loss = pred_tensor.new_tensor(float(bbox_iou_error + bbox_l1))
        soft_pred_box = self._soft_box_from_logits(pred_mask_logits)
        soft_gt_box = self._gt_box_tensor(gt_box, soft_pred_box.device, soft_pred_box.dtype)
        soft_bbox_iou = self._soft_box_iou(soft_pred_box, soft_gt_box)
        soft_bbox_l1 = self._normalized_soft_box_l1(soft_pred_box, soft_gt_box, width, height)
        soft_bbox_loss = (1.0 - soft_bbox_iou) + soft_bbox_l1
        return {
            "mask_loss": bce + dice,
            "bce_loss": bce,
            "dice_loss": dice,
            "bbox_loss": bbox_loss,
            "bbox_iou": pred_tensor.new_tensor(float(1.0 - bbox_iou_error)),
            "bbox_l1": pred_tensor.new_tensor(float(bbox_l1)),
            "soft_bbox_loss": soft_bbox_loss.to(pred_tensor.dtype),
            "soft_bbox_iou": soft_bbox_iou.to(pred_tensor.dtype),
            "soft_bbox_l1": soft_bbox_l1.to(pred_tensor.dtype),
        }

    def _clip_prompt_box(self, box: Optional[Sequence[int]], image_size: Tuple[int, int]) -> Optional[List[int]]:
        if not box or len(box) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        except Exception:
            return None
        height, width = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            return None
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1:
            x1 = max(0, min(x1, width - 2))
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y1 = max(0, min(y1, height - 2))
            y2 = min(height - 1, y1 + 1)
        return [int(x1), int(y1), int(x2), int(y2)]

    def _clip_prompt_points(self, points: Optional[Sequence[Sequence[int]]], image_size: Tuple[int, int]) -> List[List[int]]:
        if not points:
            return []
        height, width = int(image_size[0]), int(image_size[1])
        clipped: List[List[int]] = []
        for point in points:
            if not point or len(point) < 2:
                continue
            try:
                x, y = int(round(float(point[0]))), int(round(float(point[1])))
            except Exception:
                continue
            clipped.append([max(0, min(x, width - 1)), max(0, min(y, height - 1))])
        return clipped

    def _raw_prompt_box_is_missing(self, box: Optional[Sequence[int]]) -> bool:
        if not box or len(box) < 4:
            return True
        try:
            return all(int(float(v)) == 0 for v in box[:4])
        except Exception:
            return True

    def _require_valid_stage1_box(self, box: Optional[Sequence[int]], image_size: Tuple[int, int]) -> List[int]:
        if self._raw_prompt_box_is_missing(box):
            raise ValueError("invalid aligned stage1_box")
        prompt_box = self._clip_prompt_box(box, image_size)
        if prompt_box is None:
            raise ValueError("invalid aligned stage1_box")
        return prompt_box

    def _is_aligned_prompt_error(self, exc: Exception) -> bool:
        message = str(exc)
        return "aligned stage1_points" in message or "aligned stage1_box" in message or "valid stage1_box" in message

    def _gt_mask_box_and_centroid_points(
        self,
        gt_mask: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[List[int], List[List[int]]]:
        """Build an exact mask box and two identical centroid seeds from a GT mask."""
        mask = torch.as_tensor(gt_mask).detach()
        while mask.ndim > 2 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.ndim != 2:
            raise ValueError(f"aligned gt_mask must be 2D, got shape={tuple(mask.shape)}")

        foreground = torch.nonzero(mask > 0, as_tuple=False)
        if foreground.numel() == 0:
            raise ValueError("aligned gt_mask contains no foreground pixels")

        ys = foreground[:, 0]
        xs = foreground[:, 1]
        prompt_box = self._clip_prompt_box(
            [
                int(xs.min().item()),
                int(ys.min().item()),
                int(xs.max().item()),
                int(ys.max().item()),
            ],
            image_size,
        )
        if prompt_box is None:
            raise ValueError("aligned gt_mask produced an invalid prompt box")

        centroid = self._clip_prompt_points(
            [[float(xs.float().mean().item()), float(ys.float().mean().item())]],
            image_size,
        )[0]
        # The existing jitter loop samples independently for each copy.
        return prompt_box, [list(centroid), list(centroid)]

    def _jitter_aligned_prompts(
        self,
        input_box: Optional[Sequence[int]],
        input_points: Optional[Sequence[Sequence[int]]],
        image_size: Tuple[int, int],
    ) -> Tuple[Optional[List[int]], List[List[int]]]:
        prompt_box = self._clip_prompt_box(input_box, image_size)
        prompt_points = self._clip_prompt_points(input_points, image_size)
        original_points = [list(point) for point in prompt_points]
        height, width = int(image_size[0]), int(image_size[1])

        box_ratio = max(float(getattr(self, "aligned_box_jitter_ratio", 0.0)), 0.0)
        if prompt_box is not None and box_ratio > 0.0:
            x1, y1, x2, y2 = prompt_box
            box_w = max(float(x2 - x1 + 1), 1.0)
            box_h = max(float(y2 - y1 + 1), 1.0)
            center_x = (float(x1) + float(x2)) * 0.5
            center_y = (float(y1) + float(y2)) * 0.5
            # A ratio r bounds every un-clipped box-edge displacement by r of
            # its original side: 0.5r from translation and 0.5r from scaling.
            # Geometry jitter is intentionally applied only once per sample.
            center_jitter = torch.empty(2, device=self.qwen_device, dtype=torch.float32).uniform_(-0.5, 0.5)
            scale_jitter = torch.empty(2, device=self.qwen_device, dtype=torch.float32).uniform_(-1.0, 1.0)
            jittered_w = max(1.0, box_w * (1.0 + float(scale_jitter[0].item()) * box_ratio))
            jittered_h = max(1.0, box_h * (1.0 + float(scale_jitter[1].item()) * box_ratio))
            center_x += float(center_jitter[0].item()) * box_w * box_ratio
            center_y += float(center_jitter[1].item()) * box_h * box_ratio
            prompt_box = self._clip_prompt_box(
                [
                    center_x - jittered_w * 0.5,
                    center_y - jittered_h * 0.5,
                    center_x + jittered_w * 0.5,
                    center_y + jittered_h * 0.5,
                ],
                image_size,
            )

        point_ratio = max(float(getattr(self, "aligned_point_jitter_ratio", 0.0)), 0.0)
        if prompt_points and point_ratio > 0.0:
            if prompt_box is not None:
                scale_x = max(float(prompt_box[2] - prompt_box[0] + 1), 1.0)
                scale_y = max(float(prompt_box[3] - prompt_box[1] + 1), 1.0)
            else:
                scale_x = max(float(width), 1.0)
                scale_y = max(float(height), 1.0)
            jittered_points: List[List[int]] = []
            for point in prompt_points:
                jitter = torch.empty(2, device=self.qwen_device, dtype=torch.float32).uniform_(-1.0, 1.0)
                jittered_points.append(
                    [
                        int(round(float(point[0]) + float(jitter[0].item()) * scale_x * point_ratio)),
                        int(round(float(point[1]) + float(jitter[1].item()) * scale_y * point_ratio)),
                    ]
                )
            prompt_points = self._clip_prompt_points(jittered_points, image_size)
            if len(prompt_points) >= 2 and prompt_points[0] == prompt_points[1]:
                # Preserve the original pair if rounding/clipping collapses the
                # independently jittered points. GT-centroid mode intentionally
                # permits the original pair itself to contain duplicate points.
                prompt_points = original_points

        return prompt_box, prompt_points

    def _sample_aligned_prompt_variant(self) -> str:
        variants = [str(v).strip().lower() for v in getattr(self, "aligned_prompt_variants", [])]
        variants = [v for v in variants if v in {"box", "points", "box_points", "embedding"}]
        if not variants:
            variants = ["box", "points", "box_points", "embedding"]
        raw_probs = list(getattr(self, "aligned_prompt_variant_probs", []))
        if len(raw_probs) != len(variants):
            raw_probs = [1.0 for _ in variants]
        probs = torch.tensor([max(float(v), 0.0) for v in raw_probs], device=self.qwen_device, dtype=torch.float32)
        if not bool(torch.isfinite(probs).all().item()) or float(probs.sum().item()) <= 0.0:
            probs = torch.ones(len(variants), device=self.qwen_device, dtype=torch.float32)
        variant = variants[int(torch.multinomial(probs, 1).item())]
        return variant

    def _require_two_prompt_points(
        self,
        points: Optional[Sequence[Sequence[int]]],
        image_size: Tuple[int, int],
        allow_duplicates: bool = False,
    ) -> List[List[int]]:
        prompt_points = self._clip_prompt_points(points, image_size)
        if len(prompt_points) < 2:
            raise ValueError("aligned stage1_points must contain at least 2 valid points")
        if not allow_duplicates and prompt_points[0] == prompt_points[1]:
            raise ValueError("aligned stage1_points first two points must be different")
        return prompt_points[:2]

    def _require_two_direct_refiner_points(
        self,
        points: Optional[Sequence[Sequence[int]]],
        image_size: Tuple[int, int],
    ) -> List[List[int]]:
        prompt_points = self._clip_prompt_points(points, image_size)
        if len(prompt_points) < 2:
            raise ValueError("direct stage1_points must contain at least 2 valid points")
        if prompt_points[0] == prompt_points[1]:
            raise ValueError("direct stage1_points first two points must be different")
        return prompt_points[:2]

    def _prepare_aligned_training_prompts(
        self,
        input_box: Optional[Sequence[int]],
        input_points: Optional[Sequence[Sequence[int]]],
        image_size: Tuple[int, int],
        allow_duplicate_points: bool = False,
    ) -> Tuple[Optional[List[int]], List[List[int]], bool, str, List[List[int]]]:
        if bool(getattr(self, "aligned_prompt_jitter_enabled", False)):
            prompt_box, prompt_points = self._jitter_aligned_prompts(input_box, input_points, image_size)
        else:
            prompt_box = self._clip_prompt_box(input_box, image_size)
            prompt_points = self._clip_prompt_points(input_points, image_size)
        refiner_points = self._require_two_prompt_points(
            prompt_points,
            image_size,
            allow_duplicates=allow_duplicate_points,
        )
        variant = self._sample_aligned_prompt_variant() if self.aligned_prompt_sampling_enabled else "box_points"
        if variant in {"box", "box_points"} and prompt_box is None:
            raise ValueError(f"aligned prompt variant '{variant}' requires a valid stage1_box")
        if variant == "embedding":
            return None, [], False, variant, refiner_points
        if variant == "box":
            return prompt_box, [], False, variant, refiner_points
        if variant == "points":
            return None, refiner_points, True, variant, refiner_points
        return prompt_box, refiner_points, True, variant, refiner_points

    def _segment_branch(
        self,
        sam_image: torch.Tensor,
        image_size: Tuple[int, int],
        gt_mask: torch.Tensor,
        gt_box: Sequence[int],
        branch_embedding: torch.Tensor,
        input_box: Optional[List[int]] = None,
        input_points: Optional[List[List[int]]] = None,
        refiner_points: Optional[List[List[int]]] = None,
        use_points: bool = True,
        mode = "aligned",
    ) -> Dict[str, Any]:
        sparse_list, native_sparse_list, dense_list, features, prompt_aux_list = self._prepare_prompt_inputs(
            sam_images=sam_image.unsqueeze(0),
            image_sizes=[image_size],
            input_boxes=[input_box or []],
            input_points=[input_points or []],
            refiner_points=[refiner_points or []],
            extra_sparse=branch_embedding.unsqueeze(0) if self.prompt_space in {"sparse", "cross_sparse", "two_stage_sparse"} else None,
            dense_prompt_embeddings=branch_embedding.unsqueeze(0) if self.prompt_space == "dense" else None,
            prompt_refiner=(self.aligned_connector if mode == "aligned" else self.direct_connector) if self.prompt_space == "two_stage_sparse" else None,
            use_points=use_points,
            mode = mode,
        )
        pred_mask = self._decode_masks(features, sparse_list, dense_list, [image_size])[0]
        pred_mask_logits = pred_mask[0, 0]
        pred_mask_binary = (torch.sigmoid(pred_mask_logits).detach().cpu().numpy() > 0.5).astype(np.uint8)
        pred_box = mask_to_box(pred_mask_binary)
        losses = self._compute_mask_bbox_losses(
            pred_mask_logits=pred_mask_logits,
            gt_mask=gt_mask,
            pred_box=pred_box,
            gt_box=gt_box,
            width=image_size[1],
            height=image_size[0],
        )
        return {
            "pred_mask_logits": pred_mask_logits,
            "pred_mask": pred_mask_binary,
            "pred_box": pred_box,
            "losses": losses,
            "native_sparse": native_sparse_list[0],
            "connector_sparse": prompt_aux_list[0].get("connector_sparse") if prompt_aux_list else None,
        }

    def stage1_forward(self, batch: Dict[str, Any], train_branch: str = "both") -> Dict[str, Any]:
        model_inputs = batch["model_inputs"]
        sam_images = batch["sam_images"].to(self.qwen_device)
        gt_masks = batch["gt_masks"]
        gt_boxes = batch["gt_boxes"]
        image_sizes = batch["image_sizes"]
        meta = batch.get("meta", [{} for _ in range(len(image_sizes))])
        branch_key = str(train_branch or "both").strip().lower()
        if branch_key not in {"both", "aligned", "direct"}:
            raise ValueError(f"Unsupported train_branch: {train_branch}")
        query_branch = "direct" if branch_key == "direct" else "aligned"
        query_hidden_batch, query_hidden_mask = self.extract_query_hidden_states(
            model_inputs,
            require_grad=self.training,
            branch=query_branch,
        )
        aligned_embedding_batch, direct_embedding_batch = self.build_connector_embeddings(
            query_hidden_batch,
            query_hidden_mask,
            branch=branch_key,
        )

        aligned_total = torch.zeros((), device=self.qwen_device, dtype=torch.float32)
        direct_total = torch.zeros((), device=self.qwen_device, dtype=torch.float32)
        aligned_predictions: List[Dict[str, Any]] = []
        direct_predictions: List[Dict[str, Any]] = []
        aligned_valid = 0
        direct_valid = 0
        aligned_skipped = 0
        direct_skipped = 0
        metric_keys = [
            "loss",
            "mask_loss",
            "bbox_loss",
            "bbox_iou",
            "bce_loss",
            "dice_loss",
            "bbox_l1",
            "soft_bbox_loss",
            "soft_bbox_iou",
            "soft_bbox_l1",
            "box_gap_loss",
            "prompt_embedding_loss",
            "prompt_embedding_valid",
        ]
        aligned_metric_lists: Dict[str, List[float]] = {key: [] for key in metric_keys}
        direct_metric_lists: Dict[str, List[float]] = {key: [] for key in metric_keys}

        for idx in range(query_hidden_batch.size(0)):
            if aligned_embedding_batch is None:
                aligned_predictions.append({"pred_mask": np.zeros(image_sizes[idx], dtype=np.uint8), "bbox": [0, 0, 0, 0], "points": [[0, 0], [0, 0]], "skip_reason": "branch_disabled"})
            else:
                try:
                    aligned_input_box = meta[idx].get("stage1_box") if idx < len(meta) else None
                    aligned_input_points = meta[idx].get("stage1_points") if idx < len(meta) else None
                    strict_aligned_prompts = self.training and branch_key == "aligned"
                    allow_duplicate_aligned_points = False
                    aligned_refiner_points = None
                    aligned_stage1_box_for_gap = None
                    if strict_aligned_prompts:
                        aligned_prompt_source = str(
                            getattr(self, "aligned_prompt_source", "stage1")
                        ).strip().lower()
                        if aligned_prompt_source == "gt_mask":
                            aligned_input_box, aligned_input_points = self._gt_mask_box_and_centroid_points(
                                gt_masks[idx],
                                image_sizes[idx],
                            )
                            allow_duplicate_aligned_points = True
                        aligned_input_box = self._require_valid_stage1_box(aligned_input_box, image_sizes[idx])
                        aligned_stage1_box_for_gap = list(aligned_input_box)
                        aligned_refiner_points = self._require_two_prompt_points(
                            aligned_input_points,
                            image_sizes[idx],
                            allow_duplicates=allow_duplicate_aligned_points,
                        )
                        aligned_input_points = aligned_refiner_points
                        aligned_prompt_variant = "box_points"
                        aligned_use_points = True
                    else:
                        if not aligned_input_box or len(aligned_input_box) < 4 or all(int(v) == 0 for v in aligned_input_box[:4]):
                            aligned_input_box = list(gt_boxes[idx])
                            aligned_input_points = []
                        aligned_input_points = aligned_input_points or []
                        aligned_prompt_variant = "box_points" if aligned_input_points else "box"
                        aligned_use_points = bool(aligned_input_points)
                    if strict_aligned_prompts and (
                        bool(getattr(self, "aligned_prompt_sampling_enabled", False))
                        or bool(getattr(self, "aligned_prompt_jitter_enabled", False))
                    ):
                        aligned_input_box, aligned_input_points, aligned_use_points, aligned_prompt_variant, aligned_refiner_points = self._prepare_aligned_training_prompts(
                            aligned_input_box,
                            aligned_refiner_points,
                            image_sizes[idx],
                            allow_duplicate_points=allow_duplicate_aligned_points,
                        )
                    else:
                        aligned_input_box = self._clip_prompt_box(aligned_input_box, image_sizes[idx])
                        aligned_input_points = self._clip_prompt_points(aligned_input_points, image_sizes[idx])
                        # Match aligned training: native prompts may include box + points,
                        # while connector refinement is queried by points only.
                        aligned_refiner_points = aligned_input_points or None
                    aligned_result = self._segment_branch(
                        sam_image=sam_images[idx],
                        image_size=image_sizes[idx],
                        gt_mask=gt_masks[idx],
                        gt_box=gt_boxes[idx],
                        branch_embedding=aligned_embedding_batch[idx],
                        input_box=aligned_input_box,
                        input_points=aligned_input_points,
                        refiner_points=aligned_refiner_points,
                        use_points=aligned_use_points,
                        mode = "aligned",
                    )
                    aligned_losses = aligned_result["losses"]
                    box_gap_loss = self._compute_box_gap_loss(
                        aligned_stage1_box_for_gap,
                        gt_boxes[idx],
                        aligned_losses["soft_bbox_iou"],
                    )
                    prompt_embedding_weight = float(getattr(self, "aligned_prompt_embedding_loss_weight", 0.0))
                    if strict_aligned_prompts and prompt_embedding_weight > 0.0:
                        prompt_embedding_loss, prompt_embedding_valid = self._compute_prompt_embedding_loss(
                            aligned_result.get("connector_sparse"),
                            meta[idx].get("gt_point") if idx < len(meta) else None,
                            aligned_refiner_points or aligned_input_points,
                            image_sizes[idx],
                        )
                    else:
                        prompt_embedding_loss = aligned_losses["mask_loss"].new_zeros(())
                        prompt_embedding_valid = aligned_losses["mask_loss"].new_zeros(())
                    aligned_losses["box_gap_loss"] = box_gap_loss
                    aligned_losses["prompt_embedding_loss"] = prompt_embedding_loss
                    aligned_losses["prompt_embedding_valid"] = prompt_embedding_valid
                    if strict_aligned_prompts:
                        aligned_loss = aligned_losses["mask_loss"]
                        aligned_loss = aligned_loss + float(getattr(self, "aligned_soft_bbox_loss_weight", 0.0)) * aligned_losses["soft_bbox_loss"]
                        aligned_loss = aligned_loss + float(getattr(self, "aligned_box_gap_loss_weight", 0.0)) * aligned_losses["box_gap_loss"]
                        aligned_loss = aligned_loss + prompt_embedding_weight * aligned_losses["prompt_embedding_loss"]
                    else:
                        aligned_loss = aligned_losses["mask_loss"] + aligned_losses["bbox_loss"]
                    aligned_total = aligned_total + aligned_loss
                    aligned_valid += 1
                    aligned_predictions.append({"pred_mask": aligned_result["pred_mask"], "bbox": aligned_result["pred_box"], "points": aligned_refiner_points or aligned_input_points or [[0, 0], [0, 0]], "prompt_variant": aligned_prompt_variant, "skip_reason": ""})
                    aligned_metric_lists["loss"].append(float(aligned_loss.detach().item()))
                    for name, tensor in aligned_losses.items():
                        aligned_metric_lists[name].append(float(tensor.detach().item()))
                except Exception as exc:
                    if self.training and branch_key == "aligned" and not self._is_aligned_prompt_error(exc):
                        raise
                    aligned_skipped += 1
                    aligned_predictions.append({"pred_mask": np.zeros(image_sizes[idx], dtype=np.uint8), "bbox": [0, 0, 0, 0], "points": [[0, 0], [0, 0]], "skip_reason": str(exc)})

            if direct_embedding_batch is None:
                direct_predictions.append({"pred_mask": np.zeros(image_sizes[idx], dtype=np.uint8), "bbox": [0, 0, 0, 0], "points": [[0, 0], [0, 0]], "skip_reason": "branch_disabled"})
            else:
                try:
                    direct_input_box = meta[idx].get("stage1_box") if idx < len(meta) else None
                    direct_input_points = meta[idx].get("stage1_points") if idx < len(meta) else None
                    direct_input_points = direct_input_points or []
                    direct_refiner_points = self._require_two_direct_refiner_points(
                        direct_input_points, image_sizes[idx]
                    )
                    direct_result = self._segment_branch(
                        sam_image=sam_images[idx],
                        image_size=image_sizes[idx],
                        gt_mask=gt_masks[idx],
                        gt_box=gt_boxes[idx],
                        branch_embedding=direct_embedding_batch[idx],
                        input_box=direct_input_box,
                        input_points=direct_input_points,
                        refiner_points=direct_refiner_points,
                        use_points=bool(direct_input_points),
                        mode = "direct",
                    )
                    # print("direct_input_box",direct_input_box,"gt_boxes",gt_boxes[idx])
                    direct_loss = direct_result["losses"]["mask_loss"] + direct_result["losses"]["bbox_loss"]
                    direct_result["losses"]["box_gap_loss"] = direct_loss.new_zeros(())
                    direct_result["losses"]["prompt_embedding_loss"] = direct_loss.new_zeros(())
                    direct_result["losses"]["prompt_embedding_valid"] = direct_loss.new_zeros(())
                    direct_total = direct_total + direct_loss
                    direct_valid += 1
                    direct_predictions.append({"pred_mask": direct_result["pred_mask"], "bbox": direct_result["pred_box"], "points": direct_input_points or [[0, 0], [0, 0]], "skip_reason": ""})
                    direct_metric_lists["loss"].append(float(direct_loss.detach().item()))
                    for name, tensor in direct_result["losses"].items():
                        direct_metric_lists[name].append(float(tensor.detach().item()))
                except Exception as exc:
                    direct_skipped += 1
                    direct_predictions.append({"pred_mask": np.zeros(image_sizes[idx], dtype=np.uint8), "bbox": [0, 0, 0, 0], "points": [[0, 0], [0, 0]], "skip_reason": str(exc)})

        return {
            "aligned_loss": (aligned_total / aligned_valid) if aligned_valid > 0 else None,
            "direct_loss": (direct_total / direct_valid) if direct_valid > 0 else None,
            "aligned_valid_count": int(aligned_valid),
            "direct_valid_count": int(direct_valid),
            "aligned_skipped_count": int(aligned_skipped),
            "direct_skipped_count": int(direct_skipped),
            "aligned_predictions": aligned_predictions,
            "direct_predictions": direct_predictions,
            "aligned_metrics": {name: float(sum(values) / max(len(values), 1)) for name, values in aligned_metric_lists.items()},
            "direct_metrics": {name: float(sum(values) / max(len(values), 1)) for name, values in direct_metric_lists.items()},
        }

    def extract_cached_query_hidden_for_inputs(
        self,
        model_inputs: Dict[str, torch.Tensor],
        branch: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.extract_query_hidden_states(model_inputs, require_grad=False, branch=branch)

    def explicit_prompt_segment(
        self,
        image: Image.Image,
        bbox: Sequence[int],
        points: Sequence[Sequence[int]],
        sam_image_size: int,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        sam_image = build_sam_image_tensor(image, sam_image_size).to(self.qwen_device)
        image_size = (image.height, image.width)
        sparse_list, _, dense_list, features = self._prepare_prompt_inputs(
            sam_images=sam_image.unsqueeze(0),
            image_sizes=[image_size],
            input_boxes=[list(bbox)],
            input_points=[[list(points[0]), list(points[1])]],
            dense_prompt_embeddings=None,
            use_points=True,
        )
        pred_mask = self._decode_masks(features, sparse_list, dense_list, [image_size])[0]
        pred_mask_logits = pred_mask[0, 0]
        pred_mask_binary = (torch.sigmoid(pred_mask_logits).detach().cpu().numpy() > 0.5).astype(np.uint8)
        result = {
            "bbox": list(bbox),
            "points": [list(points[0]), list(points[1])],
            "pred_mask": pred_mask_binary,
            "pred_mask_logits": pred_mask_logits,
            "mask_iou": 0.0,
            "bbox_iou": 0.0,
        }
        if gt_mask is not None:
            result["mask_iou"] = compute_iou(pred_mask_binary, gt_mask)[2]
            result["bbox_iou"] = compute_box_iou(mask_to_box(pred_mask_binary), mask_to_box(gt_mask))
        return result

    def segment_with_aligned_embedding(
        self,
        image: Image.Image,
        bbox: Sequence[int],
        aligned_embedding: torch.Tensor,
        gt_box: Sequence[int],
        gt_mask: torch.Tensor,
        sam_image_size: int,
        points: Optional[Sequence[Sequence[int]]] = None,
    ) -> Dict[str, Any]:
        sam_image = build_sam_image_tensor(image, sam_image_size).to(self.qwen_device)
        image_size = (image.height, image.width)
        result = self._segment_branch(
            sam_image=sam_image,
            image_size=image_size,
            gt_mask=gt_mask,
            gt_box=gt_box,
            branch_embedding=aligned_embedding,
            input_box=list(bbox),
            input_points=[list(point) for point in points] if points else None,
            use_points=bool(points),
        )
        return result

    def segment_with_direct_embedding(
        self,
        image: Image.Image,
        direct_embedding: torch.Tensor,
        gt_box: Sequence[int],
        gt_mask: torch.Tensor,
        sam_image_size: int,
    ) -> Dict[str, Any]:
        sam_image = build_sam_image_tensor(image, sam_image_size).to(self.qwen_device)
        image_size = (image.height, image.width)
        result = self._segment_branch(
            sam_image=sam_image,
            image_size=image_size,
            gt_mask=gt_mask,
            gt_box=gt_box,
            branch_embedding=direct_embedding,
            input_box=None,
            input_points=None,
            use_points=False,
            mode="direct",
        )
        return result


