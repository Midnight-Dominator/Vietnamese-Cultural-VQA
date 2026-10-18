/* Hỏi đáp ảnh văn hóa Việt Nam: giao diện gọi API /api/v1 của backend FastAPI.
   Theo docs/api_contract.md: gửi multipart (file, question, model_id), không tự đặt Content-Type,
   khóa nút gửi khi đang chờ, không tự động thử lại khi lỗi mạng hoặc quá hạn, giữ lịch sử trong RAM. */
(() => {
  "use strict";

  // Mở trực tiếp từ Live Server (5500) hoặc Vite (5173) thì trỏ API về backend; còn lại dùng cùng origin.
  const params = new URLSearchParams(window.location.search);
  const configuredBase = window.VQA_API_BASE ?? params.get("api");
  const devPorts = ["5500", "5173"];
  const base = configuredBase ?? (devPorts.includes(window.location.port) ? "http://127.0.0.1:8000" : "");
  const API = base.replace(/\/+$/, "") + "/api/v1";

  const MAX_BYTES = 10 * 1024 * 1024;
  const MAX_QUESTION = 2000;
  const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const POLL_MS = 2000;

  const MODEL_BLURBS = {
    scratch: "ResNet101 + ViT5, tự huấn luyện hai giai đoạn (10 epoch).",
    finetuned: "Vintern-1B-v2 (InternViT + Qwen2) đã fine-tune, khoảng 0,9 tỷ tham số.",
  };
  const SAMPLE_QUESTIONS = [
    "Đây là gì?",
    "Hình ảnh này thể hiện điều gì?",
    "Đây là lễ hội hay hoạt động gì?",
  ];

  const state = {
    models: [],
    modelsSignature: "",
    selected: null,
    file: null,
    previewUrl: null,
    busy: false,
    turns: [],
    nextTurnId: 1,
    modelNodes: new Map(),
    pollTimer: null,
    tickTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  const ui = {
    status: $("server-status"),
    plate: $("plate"),
    fileInput: $("file-input"),
    drop: $("drop"),
    preview: $("preview"),
    previewImg: $("preview-img"),
    previewName: $("preview-name"),
    removeImage: $("remove-image"),
    imageError: $("image-error"),
    modelList: $("model-list"),
    modelNote: $("model-note"),
    empty: $("empty"),
    samples: $("samples"),
    thread: $("thread"),
    clearThread: $("clear-thread"),
    form: $("ask-form"),
    question: $("question"),
    counter: $("counter"),
    formError: $("form-error"),
    submit: $("submit"),
  };

  const el = (tag, props = {}, ...children) => {
    const node = document.createElement(tag);
    Object.assign(node, props);
    if (children.length) node.append(...children);
    return node;
  };

  const formatSeconds = (ms) =>
    (ms / 1000).toLocaleString("vi-VN", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " giây";

  // ------------------------------------------------------------------ gọi API
  class ApiError extends Error {
    constructor(code, message, status) {
      super(message);
      this.code = code;
      this.status = status;
    }
  }

  async function request(path, options) {
    let response;
    try {
      response = await fetch(API + path, options);
    } catch (cause) {
      throw new ApiError("NETWORK", "Không kết nối được tới máy chủ. Hãy kiểm tra backend đã chạy chưa.", 0);
    }
    let payload = null;
    try {
      payload = await response.json();
    } catch (cause) {
      payload = null; // phản hồi không phải JSON
    }
    if (!response.ok) {
      const detail = payload && payload.error;
      throw new ApiError(
        (detail && detail.code) || "HTTP_" + response.status,
        (detail && detail.message) || "Máy chủ trả về lỗi " + response.status + ".",
        response.status,
      );
    }
    return payload;
  }

  // ------------------------------------------------------------------ trạng thái máy chủ
  function setServerStatus(ok, text) {
    ui.status.textContent = text;
    ui.status.classList.toggle("is-ok", ok);
    ui.status.classList.toggle("is-down", !ok);
  }

  async function checkHealth() {
    try {
      await request("/health");
      setServerStatus(true, "Máy chủ đang chạy");
    } catch (err) {
      setServerStatus(false, "Không kết nối được máy chủ");
    }
  }

  // ------------------------------------------------------------------ mô hình
  const isUsable = (model) => model.available || model.load_state === "ready";

  function describeState(model) {
    if (!isUsable(model)) return { text: "Thiếu file trọng số", cls: "is-error" };
    switch (model.load_state) {
      case "ready": return { text: "Sẵn sàng", cls: "is-ready" };
      case "loading": return { text: "Đang nạp", cls: "is-loading" };
      case "error": return { text: "Lỗi khi nạp", cls: "is-error" };
      default: return { text: "Chưa nạp", cls: "" };
    }
  }

  function ensureSelection() {
    const current = state.models.find((m) => m.id === state.selected);
    if (current && isUsable(current)) return;
    const first = state.models.find(isUsable);
    state.selected = first ? first.id : null;
  }

  function renderModels() {
    state.modelNodes.clear();
    ui.modelList.replaceChildren();
    for (const model of state.models) {
      const input = el("input", { type: "radio", name: "model", value: model.id });
      input.addEventListener("change", () => {
        state.selected = model.id;
        syncModels();
      });
      const name = el("span", { className: "model-name", textContent: model.name });
      const stateNode = el("span", { className: "model-state" });
      const blurb = el("span", { className: "model-blurb", textContent: MODEL_BLURBS[model.id] || "" });
      const label = el("label", { className: "model" }, input, el("span", { className: "model-body" }, name, stateNode, blurb));
      ui.modelList.append(label);
      state.modelNodes.set(model.id, { label, input, stateNode });
    }
    syncModels();
  }

  // Cập nhật lớp CSS, trạng thái và ghi chú mà không dựng lại DOM (giữ focus bàn phím).
  function syncModels() {
    for (const model of state.models) {
      const nodes = state.modelNodes.get(model.id);
      if (!nodes) continue;
      const usable = isUsable(model);
      const info = describeState(model);
      nodes.input.checked = state.selected === model.id;
      nodes.input.disabled = !usable || state.busy;
      nodes.label.classList.toggle("is-selected", state.selected === model.id);
      nodes.label.classList.toggle("is-disabled", !usable);
      nodes.stateNode.textContent = info.text;
      nodes.stateNode.className = "model-state " + info.cls;
    }
    const selected = state.models.find((m) => m.id === state.selected);
    ui.modelNote.classList.remove("is-warning");
    if (!state.models.length) {
      ui.modelNote.hidden = false;
      ui.modelNote.classList.add("is-warning");
      ui.modelNote.textContent = "Chưa lấy được danh sách mô hình từ máy chủ.";
    } else if (!selected) {
      ui.modelNote.hidden = false;
      ui.modelNote.classList.add("is-warning");
      ui.modelNote.textContent =
        "Chưa có mô hình nào sẵn sàng. Đặt file trọng số vào backend/model_artifacts/ rồi tải lại trang.";
    } else if (selected.load_state !== "ready") {
      ui.modelNote.hidden = false;
      ui.modelNote.textContent = "Lần hỏi đầu tiên với mô hình này sẽ lâu hơn vì phải nạp mô hình vào bộ nhớ.";
    } else {
      ui.modelNote.hidden = true;
    }
  }

  async function refreshModels() {
    try {
      const data = await request("/models");
      const signature = JSON.stringify(data.models);
      if (signature !== state.modelsSignature) {
        state.modelsSignature = signature;
        state.models = data.models;
        ensureSelection();
        renderModels();
      }
    } catch (err) {
      state.modelsSignature = "";
      state.models = [];
      state.selected = null;
      renderModels();
      if (err.code === "NETWORK") setServerStatus(false, "Không kết nối được máy chủ");
    }
  }

  // Hỏi lại danh sách mô hình khi đang chờ trả lời hoặc có mô hình đang nạp, để hiển thị "Đang nạp".
  function schedulePoll() {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
    const loading = state.models.some((m) => m.load_state === "loading");
    if (!state.busy && !loading) return;
    state.pollTimer = setTimeout(async () => {
      await refreshModels();
      schedulePoll();
    }, POLL_MS);
  }

  // ------------------------------------------------------------------ ảnh
  function showImageError(message) {
    ui.imageError.textContent = message;
    ui.imageError.hidden = false;
  }

  function clearImageError() {
    ui.imageError.hidden = true;
    ui.imageError.textContent = "";
  }

  function setFile(file) {
    clearImageError();
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      showImageError("Chỉ nhận ảnh JPEG, PNG hoặc WEBP.");
      return;
    }
    if (file.size === 0) {
      showImageError("File ảnh rỗng. Hãy chọn ảnh khác.");
      return;
    }
    if (file.size > MAX_BYTES) {
      showImageError("Ảnh lớn hơn 10 MB. Hãy chọn ảnh nhỏ hơn.");
      return;
    }
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    ui.previewImg.src = state.previewUrl;
    ui.previewName.textContent = file.name;
    ui.preview.hidden = false;
    ui.drop.hidden = true;
    ui.previewName.hidden = false;
    ui.removeImage.hidden = false;
  }

  function pickAnotherImage() {
    ui.fileInput.value = "";
    ui.fileInput.click();
  }

  // ------------------------------------------------------------------ hỏi đáp
  function paintTurn(turn) {
    const { li, answer, meta } = turn.nodes;
    li.className = "turn is-" + turn.status;
    if (turn.status === "pending") {
      turn.nodes.elapsed = el("span", { textContent: " 0,0 giây" });
      turn.nodes.elapsed.setAttribute("aria-hidden", "true");
      answer.replaceChildren(el("span", { className: "dots", textContent: "Đang trả lời" }), turn.nodes.elapsed);
      meta.textContent = turn.modelName;
    } else if (turn.status === "done") {
      answer.textContent = turn.answer;
      meta.textContent = turn.modelName + ", " + formatSeconds(turn.elapsedMs);
    } else {
      answer.textContent = turn.error;
      meta.textContent = turn.modelName + ", mã lỗi " + turn.errorCode;
    }
  }

  function buildTurn(turn) {
    const li = el("li", { className: "turn" });
    const question = el("p", { className: "turn-q", textContent: turn.question });
    const answer = el("p", { className: "turn-a" });
    const meta = el("p", { className: "turn-meta" });
    li.append(question, answer, meta);
    turn.nodes = { li, answer, meta, elapsed: null };
    paintTurn(turn);
    return li;
  }

  function updateThreadChrome() {
    const hasTurns = state.turns.length > 0;
    ui.empty.hidden = hasTurns;
    ui.clearThread.hidden = !hasTurns || state.busy;
  }

  function friendlyError(err) {
    if (err.code === "INFERENCE_TIMEOUT") {
      return err.message + " Hãy thử lại sau hoặc chọn mô hình nhẹ hơn.";
    }
    return err.message;
  }

  function setBusy(busy) {
    state.busy = busy;
    ui.submit.disabled = busy;
    ui.submit.textContent = busy ? "Đang hỏi" : "Hỏi mô hình";
    syncModels();
    updateThreadChrome();
    schedulePoll();
  }

  function startTicker(turn) {
    const started = performance.now();
    clearInterval(state.tickTimer);
    state.tickTimer = setInterval(() => {
      if (turn.nodes.elapsed) turn.nodes.elapsed.textContent = " " + formatSeconds(performance.now() - started);
    }, 250);
    return started;
  }

  async function ask(event) {
    event.preventDefault();
    if (state.busy) return;
    ui.formError.hidden = true;
    clearImageError();

    const question = ui.question.value.trim();
    const model = state.models.find((m) => m.id === state.selected);
    if (!model) {
      ui.formError.textContent = "Chưa có mô hình nào sẵn sàng để hỏi.";
      ui.formError.hidden = false;
      return;
    }
    if (!state.file) {
      showImageError("Hãy chọn một ảnh trước khi hỏi.");
      ui.fileInput.focus();
      return;
    }
    if (!question) {
      ui.formError.textContent = "Hãy nhập câu hỏi về bức ảnh.";
      ui.formError.hidden = false;
      ui.question.focus();
      return;
    }

    const turn = { id: state.nextTurnId++, question, modelName: model.name, status: "pending" };
    state.turns.push(turn);
    ui.thread.append(buildTurn(turn));
    turn.nodes.li.scrollIntoView({ block: "nearest" });
    setBusy(true);
    const started = startTicker(turn);

    const body = new FormData();
    body.append("file", state.file);
    body.append("question", question);
    body.append("model_id", model.id);

    try {
      const data = await request("/predict", { method: "POST", body });
      turn.status = "done";
      turn.answer = data.answer;
      ui.question.value = "";
      updateCounter();
    } catch (err) {
      turn.status = "error";
      turn.error = friendlyError(err);
      turn.errorCode = err.code || "UNKNOWN";
    } finally {
      clearInterval(state.tickTimer);
      turn.elapsedMs = performance.now() - started;
      paintTurn(turn);
      setBusy(false);
      await refreshModels();
      schedulePoll();
      ui.question.focus();
    }
  }

  function updateCounter() {
    const length = ui.question.value.length;
    const near = length >= MAX_QUESTION * 0.9;
    ui.counter.textContent = near ? length + " / " + MAX_QUESTION : "";
    ui.counter.classList.toggle("is-near-limit", length >= MAX_QUESTION);
  }

  // ------------------------------------------------------------------ sự kiện
  function bindEvents() {
    ui.fileInput.addEventListener("change", () => setFile(ui.fileInput.files && ui.fileInput.files[0]));
    ui.removeImage.addEventListener("click", pickAnotherImage);

    for (const type of ["dragenter", "dragover"]) {
      ui.plate.addEventListener(type, (event) => {
        event.preventDefault();
        ui.plate.classList.add("is-dragging");
      });
    }
    for (const type of ["dragleave", "drop"]) {
      ui.plate.addEventListener(type, (event) => {
        event.preventDefault();
        ui.plate.classList.remove("is-dragging");
      });
    }
    ui.plate.addEventListener("drop", (event) => {
      const files = event.dataTransfer && event.dataTransfer.files;
      setFile(files && files[0]);
    });

    ui.form.addEventListener("submit", ask);
    ui.question.addEventListener("input", () => {
      ui.formError.hidden = true;
      updateCounter();
    });
    // Enter gửi, Shift+Enter xuống dòng; bỏ qua Enter dùng để chốt chữ khi đang gõ Telex/VNI.
    ui.question.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
        event.preventDefault();
        ui.form.requestSubmit();
      }
    });

    ui.clearThread.addEventListener("click", () => {
      state.turns = [];
      ui.thread.replaceChildren();
      updateThreadChrome();
    });

    for (const text of SAMPLE_QUESTIONS) {
      const button = el("button", { type: "button", className: "sample", textContent: text });
      button.addEventListener("click", () => {
        ui.question.value = text;
        ui.formError.hidden = true;
        updateCounter();
        ui.question.focus();
      });
      ui.samples.append(el("li", {}, button));
    }
  }

  async function init() {
    bindEvents();
    updateThreadChrome();
    await checkHealth();
    await refreshModels();
    schedulePoll();
  }

  init();
})();
