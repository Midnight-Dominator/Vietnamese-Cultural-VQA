"""Bộ quản lý và điều phối các Model AI (Model Manager).

Quy ước cốt lõi (tránh tranh chấp tài nguyên và crash server):

1. Một tiến trình chỉ chạy MỘT thao tác AI tại một thời điểm (nạp, đổi, giải phóng, suy luận
   dùng chung một asyncio.Lock). Chạy backend với đúng 1 worker.
2. Model đã nạp được giữ lại; chỉ unload + load khi client yêu cầu model_id khác.
   Tại một thời điểm RAM/VRAM chỉ chứa một model.
3. Timeout:
   - Chờ lượt trong hàng đợi tối đa `timeout_seconds`; quá hạn thì bỏ yêu cầu (TimeoutError),
     không để chạy ngầm về sau.
   - Suy luận tối đa `timeout_seconds` kể từ lúc bắt đầu chạy. Thời gian nạp model không tính.
4. Timeout HTTP hoặc client ngắt kết nối KHÔNG làm dừng thread AI đang chạy. Khi đó manager
   giữ khóa cho tới khi thread kết thúc thật sự rồi mới trả khóa, nên không có hai thao tác
   AI chồng nhau. Worker dùng bản sao ảnh riêng và tự đóng bản sao đó, vì vậy route có thể
   đóng ảnh gốc ngay khi predict() thoát.
5. Module này chỉ dùng thư viện chuẩn; cấu hình và adapter thật được import lười trong
   build_default_manager().
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.ai.base import BaseVIVQAModel, LoadState, MockVIVQAModel

logger = logging.getLogger("app.ai.model_manager")


@dataclass
class ModelEntry:
    """Mô tả một model trong registry. Adapter chỉ được tạo khi cần (factory)."""

    model_id: str
    name: str
    factory: Callable[[], BaseVIVQAModel]
    instance: Optional[BaseVIVQAModel] = None
    state: LoadState = LoadState.UNLOADED


class _LockGuard:
    """Giữ asyncio.Lock và cho phép chuyển quyền trả khóa sang worker khi bị hủy/timeout."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock
        self._handed_off = False

    def hand_off(self, future: "asyncio.Future[Any]") -> None:
        """Worker còn chạy: khóa chỉ được trả khi worker kết thúc."""
        self._handed_off = True
        future.add_done_callback(self._release_after_worker)

    def release(self) -> None:
        if not self._handed_off:
            self._lock.release()

    def _release_after_worker(self, future: "asyncio.Future[Any]") -> None:
        try:
            if not future.cancelled() and future.exception() is not None:
                logger.warning("Worker AI kết thúc muộn với lỗi: %r", future.exception())
        finally:
            self._lock.release()


class ModelManager:
    """Điều phối nạp, chuyển đổi và suy luận các mô hình AI VQA."""

    def __init__(
        self,
        entries: Sequence[ModelEntry],
        *,
        timeout_seconds: float = 180.0,
        preload: str = "",
    ) -> None:
        if not entries:
            raise ValueError("ModelManager cần ít nhất một model.")
        self._entries: Dict[str, ModelEntry] = {e.model_id: e for e in entries}
        self._timeout = float(timeout_seconds)
        self._preload = preload
        self._lock: Optional[asyncio.Lock] = None
        self._initialized = False
        self._current_model_id: Optional[str] = None
        self._preload_task: Optional["asyncio.Task[None]"] = None
        self._factory_lock = threading.Lock()

    # ------------------------------------------------------------------ trạng thái
    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def current_model_id(self) -> Optional[str]:
        return self._current_model_id

    def _get_instance(self, entry: ModelEntry) -> BaseVIVQAModel:
        with self._factory_lock:
            if entry.instance is None:
                entry.instance = entry.factory()
            return entry.instance

    def _is_available(self, entry: ModelEntry) -> bool:
        """Model có thể phục vụ: đã READY hoặc đủ file trọng số trên đĩa."""
        if entry.state is LoadState.READY:
            return True
        try:
            check = getattr(self._get_instance(entry), "artifacts_present", None)
            return bool(check()) if callable(check) else True
        except Exception:
            logger.exception("Không kiểm tra được artifact của model '%s'", entry.model_id)
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": entry.model_id,
                "name": entry.name,
                "available": self._is_available(entry),
                "load_state": entry.state.value,
            }
            for entry in self._entries.values()
        ]

    # ------------------------------------------------------------------ vòng đời
    async def startup(self) -> None:
        self._lock = asyncio.Lock()
        self._initialized = True
        logger.info("ModelManager đã sẵn sàng (models: %s).", ", ".join(self._entries))
        if self._preload:
            self._preload_task = asyncio.create_task(self._run_preload(self._preload))

    async def _run_preload(self, model_id: str) -> None:
        entry = self._entries.get(model_id)
        if entry is None or self._lock is None:
            logger.warning("PRELOAD_MODEL='%s' không tồn tại.", model_id)
            return
        guard: Optional[_LockGuard] = None
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._timeout)
            guard = _LockGuard(self._lock)
            await self._ensure_loaded(entry, guard)
            logger.info("Đã nạp sẵn model '%s'.", model_id)
        except Exception:
            logger.exception("Không nạp sẵn được model '%s'", model_id)
        finally:
            if guard is not None:
                guard.release()

    async def shutdown(self) -> None:
        self._initialized = False
        if self._preload_task is not None and not self._preload_task.done():
            self._preload_task.cancel()
        if self._lock is None:
            return
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=60)
        except asyncio.TimeoutError:
            logger.error("Worker AI chưa kết thúc sau 60s, bỏ qua bước unload khi tắt.")
            return
        try:
            if self._current_model_id is not None:
                entry = self._entries[self._current_model_id]
                await asyncio.to_thread(self._unload_entry, entry)
                self._current_model_id = None
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ worker (chạy trong thread)
    def _unload_entry(self, entry: ModelEntry) -> None:
        try:
            if entry.instance is not None:
                entry.instance.unload()
        except Exception:
            logger.exception("Lỗi khi unload model '%s'", entry.model_id)
        entry.state = LoadState.UNLOADED

    def _switch_worker(self, old: Optional[ModelEntry], new: ModelEntry) -> None:
        """Giải phóng model cũ rồi nạp model mới. Luôn chạy khi đang giữ khóa."""
        if old is not None:
            self._unload_entry(old)
            self._current_model_id = None
        new.state = LoadState.LOADING
        try:
            self._get_instance(new).load()
        except BaseException:
            new.state = LoadState.ERROR
            logger.error("Nạp model '%s' thất bại.", new.model_id)
            raise
        new.state = LoadState.READY
        self._current_model_id = new.model_id

    @staticmethod
    def _infer_worker(model: BaseVIVQAModel, image: Any, question: str) -> str:
        try:
            return model.predict(image, question)
        finally:
            try:
                image.close()
            except Exception:
                logger.exception("Không đóng được bản sao ảnh của worker.")

    # ------------------------------------------------------------------ điều phối
    async def _ensure_loaded(self, entry: ModelEntry, guard: _LockGuard) -> None:
        """Đảm bảo `entry` đang READY. Gọi khi đã giữ khóa; không tính vào timeout suy luận."""
        if self._current_model_id == entry.model_id and entry.state is LoadState.READY:
            return
        old = self._entries.get(self._current_model_id) if self._current_model_id else None
        entry.state = LoadState.LOADING
        future = asyncio.ensure_future(asyncio.to_thread(self._switch_worker, old, entry))
        try:
            await asyncio.shield(future)
        except BaseException:
            if not future.done():
                guard.hand_off(future)
            raise

    async def predict(self, model_id: str, image: Any, question: str) -> str:
        """Chạy VQA. Raise TimeoutError khi quá hạn chờ lượt hoặc quá hạn suy luận."""
        if not self._initialized or self._lock is None:
            raise RuntimeError("ModelManager chưa được khởi tạo.")
        entry = self._entries.get(model_id)
        if entry is None:
            raise ValueError(f"Model ID '{model_id}' không tồn tại.")

        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            logger.warning("Loại bỏ request: chờ quá %ss trong hàng đợi.", self._timeout)
            raise TimeoutError("Request quá hạn trong hàng đợi.") from exc

        guard = _LockGuard(self._lock)
        try:
            await self._ensure_loaded(entry, guard)

            model = self._get_instance(entry)
            worker_image = image.copy()  # bản sao riêng cho worker; worker tự đóng
            future = asyncio.ensure_future(
                asyncio.to_thread(self._infer_worker, model, worker_image, question)
            )
            try:
                return await asyncio.wait_for(asyncio.shield(future), timeout=self._timeout)
            except BaseException as exc:
                if not future.done():
                    # Thread vẫn đang chạy: giữ khóa tới khi nó xong (timeout hoặc client ngắt).
                    guard.hand_off(future)
                if isinstance(exc, asyncio.TimeoutError):
                    logger.warning("Suy luận quá %ss; worker tiếp tục tới khi xong.", self._timeout)
                    raise TimeoutError("Suy luận vượt quá thời gian cho phép.") from exc
                raise
        finally:
            guard.release()


# ---------------------------------------------------------------------- mặc định của ứng dụng
MODEL_DISPLAY_NAMES = {
    "scratch": "Model tự xây",
    "finetuned": "Model fine-tune",
}


def build_default_manager(settings: Any = None) -> ModelManager:
    """Tạo manager với 2 model của dự án: scratch (ResNet101 + ViT5) và finetuned (Vintern-1B-v2)."""
    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    if settings.USE_MOCK_MODELS:
        logger.warning("USE_MOCK_MODELS=true: dùng model mô phỏng, KHÔNG phải model thật.")
        entries = [
            ModelEntry("scratch", MODEL_DISPLAY_NAMES["scratch"], lambda: MockVIVQAModel("scratch", load_delay=2.0, predict_delay=1.0)),
            ModelEntry("finetuned", MODEL_DISPLAY_NAMES["finetuned"], lambda: MockVIVQAModel("finetuned", load_delay=3.0, predict_delay=2.0)),
        ]
    else:

        def make_scratch() -> BaseVIVQAModel:
            from app.ai.scratch_model import ScratchVIVQAModel

            return ScratchVIVQAModel()

        def make_finetuned() -> BaseVIVQAModel:
            from app.ai.finetuned_model import FinetunedVIVQAModel

            return FinetunedVIVQAModel()

        entries = [
            ModelEntry("scratch", MODEL_DISPLAY_NAMES["scratch"], make_scratch),
            ModelEntry("finetuned", MODEL_DISPLAY_NAMES["finetuned"], make_finetuned),
        ]

    return ModelManager(
        entries,
        timeout_seconds=settings.INFERENCE_TIMEOUT_SECONDS,
        preload=settings.PRELOAD_MODEL,
    )
