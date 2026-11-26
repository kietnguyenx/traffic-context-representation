/**
 * Entry Point Chính (Application Controller)
 * Điều phối kết nối giữa State Store, REST API, WebSocket Binary Stream và toàn bộ UI Components.
 */

import { store } from "./state.js";
import {
  fetchSystemStatus,
  uploadVideoFile,
  clearVideoOnServer,
  syncTogglesToServer,
  stopAnalysisOnServer,
} from "./api.js";
import { AnalysisStreamClient } from "./stream.js";
import { renderHeaderStatus } from "./ui/header.js";
import {
  initUploader,
  setUploaderEmpty,
  setUploaderUploading,
  setUploaderReady,
  showNoVideoAlert,
  hideNoVideoAlert,
} from "./ui/uploader.js";
import {
  showStandbyViewport,
  showPosterPreview,
  renderLiveFrameBlob,
  showAnalyzedVideoPlayer,
  updateProgress,
} from "./ui/viewport.js";
import { renderCockpitTelemetry } from "./ui/cockpit.js";
import { renderJsonLog } from "./ui/jsonLog.js";
import { showToast } from "./ui/toast.js";

const btnStart = document.getElementById("btn-start-analysis");
const toggleEls = {
  show_boxes: document.getElementById("toggle-boxes"),
  show_drivable: document.getElementById("toggle-drivable"),
  show_lanes: document.getElementById("toggle-lanes"),
  show_graph: document.getElementById("toggle-graph"),
};

const logCardEl = document.getElementById("context-log-card");
const controlCardEl = document.querySelector(".control-card");
const rightColEl = document.querySelector(".col-right");

let streamClient = null;

/**
 * Căn bằng khít mép đáy của Nhật Ký JSON (trái) với Cụm Đồng Hồ Ribbon Dock (phải)
 * Đảm bảo trang luôn gọn trong 1 khung màn hình Laptop (Zero Page Scroll).
 */
function syncLeftLogHeight() {
  if (!logCardEl || !controlCardEl || !rightColEl || window.innerWidth <= 1024) {
    if (logCardEl) logCardEl.style.maxHeight = "";
    return;
  }
  const rightH = rightColEl.getBoundingClientRect().height;
  const ctrlH = controlCardEl.getBoundingClientRect().height;
  const targetH = Math.max(220, Math.floor(rightH - ctrlH - 10));
  logCardEl.style.height = `${targetH}px`;
  logCardEl.style.maxHeight = `${targetH}px`;
}

function setStartButtonState(isRunning) {
  store.isAnalyzing = isRunning;
  if (isRunning) {
    btnStart.textContent = "STOP ANALYSIS";
    btnStart.classList.add("is-running");
  } else {
    btnStart.textContent = "START ANALYSIS";
    btnStart.classList.remove("is-running");
  }
}

function renderStandbyState(isWarning = false) {
  showStandbyViewport(isWarning);
  renderCockpitTelemetry({
    speed: "--",
    lead_dist: "--",
    ttc: "--",
    risk: "STANDBY",
    signal: isWarning ? "Clear" : "Standby",
    ped_hazard: "0 Sidewalk (Safe)",
    lead_id: null,
  });

  if (isWarning) {
    renderJsonLog({
      status: "INPUT_REQUIRED",
      warning: "No dashcam video stream currently loaded.",
      action: "Drag and drop a video above and click 'START ANALYSIS'.",
    });
  } else {
    renderJsonLog({
      status: "STANDBY",
      message: "Waiting for dashcam video stream input...",
      models: {
        detector: "YOLOv12-L",
        segmenter: `SegFormer ${store.models?.segformer?.backbone || "MiT-B2"}`,
      },
      supported_codecs: ["H.264", "MPEG-4", "HEVC"],
    });
  }
  requestAnimationFrame(syncLeftLogHeight);
}

function renderVideoReadyState(meta) {
  store.hasVideo = true;
  store.videoMeta = meta;
  setUploaderReady(meta.file_name);
  showPosterPreview(meta.poster_b64);
  renderCockpitTelemetry({
    speed: 0,
    lead_dist: "--",
    ttc: "--",
    risk: "SAFE",
    signal: "Ready",
    ped_hazard: "0 Sidewalk (Safe)",
    lead_id: null,
  });
  renderJsonLog({
    status: "VIDEO_READY",
    file_name: meta.file_name,
    resolution: meta.resolution,
    fps: meta.fps,
    total_frames: meta.total_frames,
    action: "Click 'START ANALYSIS' to run the perception pipeline",
  });
  requestAnimationFrame(syncLeftLogHeight);
}

async function handleFileSelected(file) {
  if (store.isAnalyzing) {
    await stopCurrentAnalysis();
  }
  store.isUploading = true;
  setUploaderUploading(file.name);
  btnStart.disabled = true;

  try {
    const res = await uploadVideoFile(file);
    renderVideoReadyState(res.video_meta);
    showToast(`Video loaded: ${res.video_meta.file_name}`, "success");
  } catch (err) {
    setUploaderEmpty();
    store.hasVideo = false;
    store.videoMeta = null;
    renderStandbyState(false);
    showToast(err.message || "Error uploading video file.", "error");
  } finally {
    store.isUploading = false;
    btnStart.disabled = false;
  }
}

async function handleClearVideo() {
  if (store.isAnalyzing) {
    await stopCurrentAnalysis();
  }
  try {
    await clearVideoOnServer();
  } catch (_) {
    // ignore
  }
  store.hasVideo = false;
  store.videoMeta = null;
  setUploaderEmpty();
  hideNoVideoAlert();
  renderStandbyState(false);
  showToast("Video removed and buffer cleared.", "info");
}

async function stopCurrentAnalysis() {
  if (streamClient) {
    streamClient.stop();
    streamClient.close();
    streamClient = null;
  }
  await stopAnalysisOnServer();
  setStartButtonState(false);
}

function startAnalysisStream() {
  if (!store.hasVideo) {
    showNoVideoAlert();
    renderStandbyState(true);
    showToast("Please upload a dashcam video before starting analysis!", "error");
    return;
  }

  hideNoVideoAlert();
  setStartButtonState(true);

  streamClient = new AnalysisStreamClient({
    onFrameBlob: (blob) => {
      renderLiveFrameBlob(blob);
    },
    onTelemetry: (msg) => {
      updateProgress(msg.current_frame, msg.total_frames, msg.out_fps, false);
      renderCockpitTelemetry(msg.telemetry);
      if (msg.context_log) {
        renderJsonLog(msg.context_log, {
          summaryStripObj: msg.telemetry?.summary_strip || null,
        });
      }
      syncLeftLogHeight();
    },
    onCompleted: (msg) => {
      setStartButtonState(false);
      const conclusion = msg.conclusion;
      if (conclusion) {
        updateProgress(
          conclusion.telemetry.total_frames,
          conclusion.telemetry.total_frames,
          conclusion.telemetry.out_fps,
          true
        );
        renderCockpitTelemetry(conclusion.telemetry);
        renderJsonLog(conclusion.conclusion_json, {
          conclusionInfo: conclusion.conclusion_info,
        });
      }
      if (msg.video_url) {
        showAnalyzedVideoPlayer(msg.video_url);
      }
      showToast("Analysis complete and trip context summarized!", "success");
      requestAnimationFrame(syncLeftLogHeight);
    },
    onCancelled: () => {
      setStartButtonState(false);
      showToast("Video analysis stopped.", "warning");
    },
    onError: (errMsg) => {
      setStartButtonState(false);
      showToast(errMsg, "error");
    },
    onClose: () => {
      setStartButtonState(false);
    },
  });

  streamClient.start();
}

function initToggles() {
  Object.entries(toggleEls).forEach(([key, inputEl]) => {
    if (!inputEl) return;
    inputEl.addEventListener("change", () => {
      store.toggles[key] = inputEl.checked;
      // Gửi tức thì qua WebSocket (tác dụng ngay frame kế tiếp nếu đang chạy) + lưu trên server
      if (streamClient) {
        streamClient.sendToggles(store.toggles);
      }
      syncTogglesToServer(store.toggles).catch(() => {});
    });
  });
}

async function bootstrap() {
  initUploader({
    onFileSelected: handleFileSelected,
    onClearClicked: handleClearVideo,
  });
  initToggles();

  btnStart.addEventListener("click", () => {
    if (store.isAnalyzing) {
      stopCurrentAnalysis();
    } else {
      startAnalysisStream();
    }
  });

  if (rightColEl && typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => syncLeftLogHeight());
    ro.observe(rightColEl);
  }
  window.addEventListener("resize", syncLeftLogHeight);

  // Tải trạng thái khởi tạo từ server
  try {
    const status = await fetchSystemStatus();
    store.models = status.models || store.models;
    renderHeaderStatus(store.models);

    if (status.toggles) {
      Object.entries(status.toggles).forEach(([k, val]) => {
        store.toggles[k] = Boolean(val);
        if (toggleEls[k]) toggleEls[k].checked = Boolean(val);
      });
    }

    if (status.has_video && status.video_meta) {
      if (status.has_analyzed_video && status.conclusion) {
        store.hasVideo = true;
        store.videoMeta = status.video_meta;
        setUploaderReady(status.video_meta.file_name);
        showAnalyzedVideoPlayer(`/api/video/analyzed?t=${Date.now()}`);
        updateProgress(
          status.conclusion.telemetry.total_frames,
          status.conclusion.telemetry.total_frames,
          status.conclusion.telemetry.out_fps,
          true
        );
        renderCockpitTelemetry(status.conclusion.telemetry);
        renderJsonLog(status.conclusion.conclusion_json, {
          conclusionInfo: status.conclusion.conclusion_info,
        });
      } else {
        renderVideoReadyState(status.video_meta);
      }
    } else {
      renderStandbyState(false);
    }
  } catch (err) {
    renderStandbyState(false);
    showToast("Waiting for FastAPI backend connection...", "warning");
  }

  requestAnimationFrame(syncLeftLogHeight);
}

bootstrap();
