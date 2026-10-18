"""Vintern-1B-v2 loading and preprocessing compatible with Minh's notebook checkpoint."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

IMAGE_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BASE_MODEL = "5CD-AI/Vintern-1B-v2"
CHECKPOINT_NAME = "best_vintern_vqa.pt"
ANSWER_MARKER = "Trả lời:"
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": False,
    "num_beams": 3,
    "repetition_penalty": 1.5,
}


def preprocess_image(image: Any):
    """Match notebook image transform: RGB, resize 448x448, ImageNet normalize."""
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    transform = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0)


@contextmanager
def _allow_item_on_meta_tensor(torch):
    """Compatibility shim used by the notebook while loading this remote-code model."""
    original_item = torch.Tensor.item

    def safe_item(self, *args, **kwargs):
        if self.device.type == "meta":
            return 0.0
        return original_item(self, *args, **kwargs)

    torch.Tensor.item = safe_item
    try:
        yield
    finally:
        torch.Tensor.item = original_item


def _ensure_flash_attn_stub() -> None:
    """Đăng ký flash_attn giả nếu máy không có, giống notebook (cell 15).

    Code remote của Vintern import flash_attn trong khối try/except; stub giúp bước kiểm tra
    import của transformers không báo thiếu thư viện. Cờ use_flash_attn vẫn bị tắt ở config.
    """
    import importlib.machinery
    import importlib.util
    import sys
    import types

    if "flash_attn" in sys.modules:
        return
    try:
        if importlib.util.find_spec("flash_attn") is not None:
            return
    except (ImportError, ValueError):
        pass

    def make(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return module

    sys.modules["flash_attn"] = make("flash_attn")
    for sub in ("flash_attn_interface", "bert_padding", "layers", "layers.rotary",
                "ops", "ops.fused_dense", "modules", "modules.mha"):
        sys.modules[f"flash_attn.{sub}"] = make(f"flash_attn.{sub}")


def _patch_transformers_compat() -> None:
    """Apply the idempotent Transformers compatibility patches from Minh's notebook."""
    from transformers import PreTrainedModel
    if not getattr(PreTrainedModel, "_all_tied_weights_patch_applied", False):
        original_init = PreTrainedModel.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        PreTrainedModel.__init__ = patched_init
        PreTrainedModel._all_tied_weights_patch_applied = True

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    except ImportError:
        return
    if getattr(Qwen2ForCausalLM, "_return_dict_dup_patch_applied", False):
        return
    original_prepare = Qwen2ForCausalLM.prepare_inputs_for_generation

    import functools

    original_prepare = Qwen2ForCausalLM.prepare_inputs_for_generation

    @functools.wraps(original_prepare)
    def patched_prepare(self, *args, **kwargs):
        inputs = original_prepare(self, *args, **kwargs)
        inputs.pop("return_dict", None)
        return inputs

    Qwen2ForCausalLM.prepare_inputs_for_generation = patched_prepare
    Qwen2ForCausalLM._return_dict_dup_patch_applied = True


def load_vintern_v2(base_model: str, checkpoint_path: str | Path, dtype: Any):
    """Load the v2 base model, then apply notebook's full model_state_dict checkpoint."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    _ensure_flash_attn_stub()
    _patch_transformers_compat()
    base = Path(base_model)
    local_only = base.is_dir()
    config = AutoConfig.from_pretrained(
        base_model, trust_remote_code=True, local_files_only=local_only
    )
    # The notebook disables flash attention because it is unavailable in its runtime.
    if hasattr(config, "use_flash_attn"):
        config.use_flash_attn = False
    if hasattr(config, "vision_config") and hasattr(config.vision_config, "use_flash_attn"):
        config.vision_config.use_flash_attn = False

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, use_fast=False, local_files_only=local_only
    )
    with _allow_item_on_meta_tensor(torch):
        model = AutoModel.from_pretrained(
            base_model,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
            _fast_init=False,
            trust_remote_code=True,
            local_files_only=local_only,
        )

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} không có khóa model_state_dict như notebook của Minh."
        )
    model.load_state_dict(state, strict=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, checkpoint


def build_inference_prompt(question: str) -> str:
    """The backend has no category/keyword metadata, so match notebook's fallback path."""
    return "<image>\n" + question


def extract_final_answer(text: str) -> str:
    text = (text or "").strip()
    marker = ANSWER_MARKER
    if marker in text:
        answer = text.split(marker, 1)[-1].strip()
        if answer:
            return answer
        before = text.split(marker, 1)[0].strip()
        return before or text
    return text
