"""Adapter cho checkpoint fine-tune Vintern-1B-v2 (best_vintern_vqa.pt).

Checkpoint do notebook rtx6000vqa-vintern.ipynb lưu: dict gồm model_state_dict (toàn bộ model,
đóng băng InternViT, huấn luyện mlp1 + Qwen2), epoch, val_loss, config. Adapter dựng model nền
Vintern-1B-v2 rồi nạp state dict đó (không phải LoRA, không cần merge).
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

from app.ai.base import BaseVIVQAModel
from app.config import get_settings
from app.errors import AppError

logger = logging.getLogger("app.ai.finetuned_model")


def _model_unavailable(message: str) -> AppError:
    return AppError(code="MODEL_UNAVAILABLE", message=message, status_code=503)


class FinetunedVIVQAModel(BaseVIVQAModel):
    """Nạp best_vintern_vqa.pt lên Vintern-1B-v2 và phục vụ theo giao diện chung."""

    model_id = "finetuned"

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        base_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_path = Path(model_path) if model_path is not None else settings.resolved_finetuned_model_path
        self.device = device if device is not None else settings.AI_DEVICE
        self.base_model = base_model if base_model is not None else settings.VINTERN_BASE
        self._checkpoint_name = settings.FINETUNED_CHECKPOINT
        self._model: Any = None
        self._tokenizer: Any = None
        self._checkpoint_metadata: dict[str, Any] = {}
        self._is_loaded = False

    def _checkpoint_file(self) -> Path:
        path = Path(self.model_path)
        if path.is_file():
            return path
        return path / self._checkpoint_name

    def artifacts_present(self) -> bool:
        """Kiểm tra sơ bộ để GET /models; không đảm bảo nạp được."""
        try:
            return self._checkpoint_file().is_file()
        except OSError:
            return False

    def load(self) -> None:
        if self._is_loaded:
            return
        checkpoint = self._checkpoint_file()
        if not checkpoint.is_file():
            logger.error("Không tìm thấy checkpoint fine-tune: %s", checkpoint)
            raise _model_unavailable(
                "Chưa có checkpoint fine-tune. Đặt best_vintern_vqa.pt vào thư mục model_artifacts/finetuned."
            )
        try:
            import torch

            from app.ai.vintern_utils import load_vintern_v2

            dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            model, tokenizer, metadata = load_vintern_v2(self.base_model, checkpoint, dtype=dtype)
            self._model = model.to(self.device).eval()
            self._tokenizer = tokenizer
            self._checkpoint_metadata = {
                "epoch": metadata.get("epoch"),
                "val_loss": metadata.get("val_loss"),
            }
            self._is_loaded = True
            logger.info("Đã nạp Vintern v2 fine-tune %s (%s)", checkpoint.name, self._checkpoint_metadata)
        except AppError:
            raise
        except Exception as exc:
            logger.exception("Không nạp được checkpoint Vintern v2 từ %s", checkpoint)
            self.unload()
            raise _model_unavailable(
                "Không nạp được model fine-tune. Kiểm tra VINTERN_BASE, checkpoint và phiên bản "
                "torch/transformers (xem log của server)."
            ) from exc

    def predict(self, image: Any, question: str) -> str:
        if not self._is_loaded:
            raise _model_unavailable("Model fine-tune chưa được nạp.")
        try:
            import torch

            from app.ai.vintern_utils import (
                GENERATION_CONFIG,
                build_inference_prompt,
                extract_final_answer,
                preprocess_image,
            )

            pixels = preprocess_image(image).to(device=self.device, dtype=next(self._model.parameters()).dtype)
            prompt = build_inference_prompt(question.strip())
            generation = dict(GENERATION_CONFIG)
            generation["pad_token_id"] = self._tokenizer.eos_token_id
            with torch.inference_mode():
                raw = self._model.chat(
                    self._tokenizer,
                    pixels,
                    prompt,
                    generation,
                    history=None,
                    return_history=False,
                )
            answer = extract_final_answer(raw)
            if not answer:
                raise ValueError("Vintern không sinh câu trả lời.")
            return answer
        except AppError:
            raise
        except Exception as exc:
            logger.exception("Suy luận Vintern v2 thất bại")
            raise AppError(
                code="INFERENCE_FAILED",
                message="Suy luận model fine-tune thất bại.",
                status_code=500,
            ) from exc

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
