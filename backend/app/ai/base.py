"""Lớp cơ sở trừu tượng cho Model AI Adapter (Base Model Interface).

Mục tiêu: quy định hợp đồng chung cho mọi adapter model AI (scratch_model và finetuned_model).

Quy ước bắt buộc:
1. Giao diện đồng bộ (Synchronous):
   - load() -> None: nạp trọng số từ artifact vào RAM/VRAM.
   - predict(image: PIL.Image.Image, question: str) -> str: tiền xử lý riêng, inference, giải mã.
   - unload() -> None: giải phóng model và thu dọn bộ nhớ RAM/VRAM (gọi lặp lại phải an toàn).
2. Các hàm trên chạy blocking (CPU/GPU bound). ModelManager chịu trách nhiệm chạy chúng
   ngoài event loop của FastAPI (asyncio.to_thread) và tuần tự hóa các lượt gọi.
3. Module này chỉ dùng thư viện chuẩn: không import torch, transformers hay pydantic.
4. Adapter có thể có thêm artifacts_present() -> bool để ModelManager biết model có sẵn
   trọng số hay chưa (hiển thị ở GET /models). Không bắt buộc.
"""

from abc import ABC, abstractmethod
from enum import Enum
import time
from typing import Any


class LoadState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class BaseVIVQAModel(ABC):
    """Giao diện chuẩn cho các adapter mô hình VQA."""

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Any, question: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        raise NotImplementedError


# --- MODEL MÔ PHỎNG: chỉ dùng khi đặt USE_MOCK_MODELS=true hoặc trong kiểm thử ---
class MockVIVQAModel(BaseVIVQAModel):
    """Model giả lập để thử giao diện và kiểm thử điều phối khi chưa có trọng số.

    Câu trả lời luôn được gắn nhãn [Mô phỏng] để không bị nhầm với kết quả của model thật.
    """

    def __init__(
        self,
        label: str = "mock",
        load_delay: float = 0.0,
        predict_delay: float = 0.0,
        fail_load: bool = False,
    ) -> None:
        self.label = label
        self.load_delay = load_delay
        self.predict_delay = predict_delay
        self.fail_load = fail_load
        self.loaded = False

    def load(self) -> None:
        time.sleep(self.load_delay)
        if self.fail_load:
            raise RuntimeError("Mock load failure")
        self.loaded = True

    def predict(self, image: Any, question: str) -> str:
        if not self.loaded:
            raise RuntimeError("Mock model chưa được nạp")
        time.sleep(self.predict_delay)
        return f"[Mô phỏng · {self.label}] Chưa dùng model thật. Câu hỏi nhận được: {question}"

    def unload(self) -> None:
        self.loaded = False
